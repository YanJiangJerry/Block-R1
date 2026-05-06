"""
R1 Evaluation: adaptive block size selection at test time via embedding routing.

Loads r1_controller_state.json produced during R1 training, which contains:
  - Q-values: per-domain optimal block sizes learned through GRPO
  - Domain embedding centroids: for zero-shot routing of unseen prompts

For each eval batch, computes prompt embeddings → cosine similarity to domain
centroids → selects best block size from nearest domain's Q-table → generates.

This file is completely standalone and does NOT modify eval.py or any existing code.

Usage:
  python3 -m torch.distributed.run --nproc_per_node=4 rl/eval/r1_eval.py \
    --model_path inclusionAI/LLaDA2.1-mini \
    --r1_controller_path checkpoints/r1_wd1_multi_domain/.../r1_controller_state.json \
    --dataset gsm8k --gen_length 256

  # Falls back to fixed --block_length if no controller is provided.
"""

import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

try:
    from datasets.features.features import _FEATURE_TYPES
    if "List" not in _FEATURE_TYPES:
        from datasets.features import Sequence
        _FEATURE_TYPES["List"] = Sequence
except (ImportError, AttributeError):
    pass

import argparse
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from rl.eval.generate import generate
from rl.trainers.cross_domain_generate import (
    BlockSizeController,
    compute_prompt_embedding,
)

from rl.eval.countdown import CTDDataset
from rl.eval.gsm8k import GSM8KDataset
from rl.eval.math500 import MATH500Dataset
from rl.eval.sudoku import SudokuDataset
from rl.eval.mbpp import MBPPDataset
from rl.eval.humaneval import HumanEvalDataset
from rl.eval.kodcode import KodCodeDataset
from rl.eval.mmlu import MMLUDataset
from rl.eval.mmlu_pro import MMLUProDataset
from rl.eval.hellaswag import HellaSwagDataset
from rl.eval.arc_c import ARCCDataset
from rl.eval.arc_e import ARCEDataset
from rl.eval.gpqa import GPQADataset
from rl.eval.knights_and_knaves import KnightsAndKnavesDataset

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "mbpp": MBPPDataset,
    "humaneval": HumanEvalDataset,
    "kodcode": KodCodeDataset,
    "mmlu": MMLUDataset,
    "mmlu_pro": MMLUProDataset,
    "hellaswag": HellaSwagDataset,
    "arc_c": ARCCDataset,
    "arc_e": ARCEDataset,
    "gpqa": GPQADataset,
    "knights_and_knaves": KnightsAndKnavesDataset,
}

from peft import PeftModel, LoraConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from rl.llada2_compat import (
    ensure_transformers_kwargs,
    load_diffusion_model,
    patch_llada2_block_causal_attention,
    patch_prepare_inputs_for_generation,
    patch_sdar_for_eval,
    set_llada2_eval_block_length,
)


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# Reuse the same sampler from eval.py
class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )
        self.shuffle = shuffle
        self.seed = seed


def _select_block_size(model, tokenizer, input_ids, controller, fallback_block_length):
    """Select block size via R1 embedding routing; fall back to fixed size if unavailable."""
    if controller is None:
        return fallback_block_length
    try:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        emb = compute_prompt_embedding(model, input_ids, pad_id)
        if emb is None:
            if dist.get_rank() == 0:
                print(
                    f"[R1 eval][WARN] compute_prompt_embedding returned None — "
                    f"using fallback block_length={fallback_block_length}"
                )
            return fallback_block_length
        batch_centroid = emb.mean(dim=0)
        return int(controller.select_block_size_by_embedding(batch_centroid))
    except Exception as e:
        if dist.get_rank() == 0:
            print(
                f"[R1 eval][WARN] routing failed ({e!r}) — "
                f"using fallback block_length={fallback_block_length}"
            )
        return int(fallback_block_length)


def r1_evaluate(
    model,
    tokenizer,
    dataloader,
    controller,
    gen_length=256,
    temperature=0.0,
    cfg_scale=0.0,
    steps=128,
    fallback_block_length=32,
    real_time_callback=None,
    dataset=None,
):
    """Evaluate with R1 adaptive block size selection per batch."""
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    # Track which block sizes were selected across all batches
    selected_block_sizes = []

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        # Align tokenization with R1 training (_R1Mixin): add_special_tokens=False, left pad.
        # Ensures prototype / centroid routing matches the embedding space used when saving r1.json.
        if controller is not None and batch.get("prompts") is not None:
            enc = tokenizer(
                text=list(batch["prompts"]),
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
        else:
            input_ids = batch["input_ids"].to(device)

        # R1: select block size via embedding routing
        block_length = _select_block_size(
            model, tokenizer, input_ids, controller, fallback_block_length
        )
        selected_block_sizes.append(block_length)
        # LLaDA2-MoE: block-causal mask must match generate()'s block_length (not just CLI fallback).
        set_llada2_eval_block_length(model, block_length)

        if dataset == "mbpp" and "test_list" in batch:
            sb = 16
            generated_texts = []
            N = input_ids.size(0)
            for start in range(0, N, sb):
                end = min(start + sb, N)
                mini_inputs = input_ids[start:end]
                out_mini = generate(
                    model, mini_inputs, tokenizer,
                    steps=steps, gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature, cfg_scale=cfg_scale,
                    remasking="low_confidence",
                )
                texts = tokenizer.batch_decode(
                    out_mini[:, -gen_length:], skip_special_tokens=False
                )
                generated_texts.extend(texts)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            out = generate(
                model, input_ids, tokenizer,
                steps=steps, gen_length=gen_length,
                block_length=block_length,
                temperature=temperature, cfg_scale=cfg_scale,
                remasking="low_confidence",
            )
            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=False
            )

        example_result = []
        for j in range(len(gt_answers)):
            result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
                "r1_block_size": block_length,
            }
            if "test_list" in batch:
                result["test_list"] = batch["test_list"][j]
            example_result.append(result)

        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"[R1 block_size={block_length}] Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

        if real_time_callback is not None:
            for sample in example_result:
                real_time_callback(sample)

    # Summary: which block size was used most often
    if selected_block_sizes and dist.get_rank() == 0:
        from collections import Counter
        counts = Counter(selected_block_sizes)
        print(f"\n[R1] Block size usage across batches: {dict(counts)}")

    avg_wall_time = sum(wall_times) / len(wall_times) if wall_times else 0.0
    dominant_bs = max(set(selected_block_sizes), key=selected_block_sizes.count) if selected_block_sizes else fallback_block_length
    return {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
        "r1_block_size_used": dominant_bs,
    }


def load_r1_controller(controller_path, gen_length=256, default_block_size=32):
    """Load R1 controller from JSON (classic per-domain or adaptive DSCB prototypes)."""
    with open(controller_path, "r") as f:
        state = json.load(f)

    if state.get("controller_kind") == "adaptive_dscb":
        from rl.trainers.r1_adaptive_proto import load_adaptive_proto_controller

        controller = load_adaptive_proto_controller(
            state,
            gen_length=gen_length,
            default_block_size=int(default_block_size),
        )
        return controller

    domains = list(state.get("q_values", {}).keys())
    candidates = [int(x) for x in state.get("candidates", [16, 32, 64, 128])]

    controller = BlockSizeController(
        domains=domains,
        block_size_candidates=candidates,
        gen_length=gen_length,
        default_block_size=int(default_block_size),
    )
    controller.load_state_dict(state)
    return controller


if __name__ == "__main__":
    init_seed(42)
    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str,
        choices=list(DATASET_MAP.keys()),
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32,
                        help="Fallback block length when R1 controller is unavailable")
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    # R1-specific arguments
    parser.add_argument("--r1_controller_path", type=str, default="",
                        help="Path to r1_controller_state.json from R1 training")
    args = parser.parse_args()

    args.diffusion_steps = args.gen_length // 2

    num_evals = {
        "gsm8k": -1, "math": -1, "countdown": -1, "mbpp": -1,
        "humaneval": -1, "sudoku": 256, "kodcode": -1, "mmlu": -1,
        "mmlu_pro": -1, "hellaswag": -1, "arc_c": -1, "arc_e": -1, "gpqa": -1,
        "knights_and_knaves": -1,
    }

    # Load R1 controller
    controller = None
    if args.r1_controller_path and os.path.isfile(args.r1_controller_path):
        controller = load_r1_controller(
            args.r1_controller_path,
            gen_length=args.gen_length,
            default_block_size=args.block_length,
        )
        if local_rank == 0:
            print(f"[R1] Controller loaded from {args.r1_controller_path}")
            print(f"[R1] Block size candidates: {controller.candidates}")
            kind = getattr(controller, "CONTROLLER_KIND", None)
            if kind == "adaptive_dscb":
                apc = getattr(controller, "active_prototype_count", lambda: 0)()
                print(f"[R1] DSCB adaptive prototypes (slot ids 0..K-1), active={apc}")
                for d in controller.domains:
                    if not d.isdigit():
                        continue
                    if not controller._slot_active(int(d)):
                        continue
                    best = controller.get_best_block_size(d)
                    print(f"[R1]   proto {d}: best_block_size={best}, Q={controller.q_values[d]}")
                has_proto = apc > 0
                print(f"[R1] Prototype embeddings loaded: {has_proto}")
            else:
                print(f"[R1] Known domains: {controller.domains}")
                for d in controller.domains:
                    best = controller.get_best_block_size(d)
                    print(f"[R1]   {d}: best_block_size={best}, Q={controller.q_values[d]}")
                has_centroids = any(
                    controller.get_domain_centroid(d) is not None for d in controller.domains
                )
                print(f"[R1] Embedding centroids available: {has_centroids}")
    else:
        if local_rank == 0:
            print(f"[R1] No controller path provided — using fixed block_length={args.block_length}")

    ensure_transformers_kwargs()
    device = torch.device(f"cuda:{local_rank}")
    model = load_diffusion_model(args.model_path, torch_dtype=torch.bfloat16, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        try:
            import inspect
            config_path = os.path.join(args.checkpoint_path, "adapter_config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            signature = inspect.signature(LoraConfig.__init__)
            valid_params = set(signature.parameters.keys())
            sanitized = {k: v for k, v in config_dict.items() if k in valid_params}
            removed = set(config_dict.keys()) - set(sanitized.keys())
            if removed and local_rank == 0:
                print(f"[Warning] Removed incompatible config keys: {removed}")
            config = LoraConfig(**sanitized)
            patch_prepare_inputs_for_generation(model)
            model = PeftModel.from_pretrained(
                model, args.checkpoint_path, config=config, torch_dtype=torch.bfloat16
            ).to(local_rank)
        except Exception as e:
            print(f"Rank {local_rank}: Auto-fix failed ({e}), trying standard load...")
            patch_prepare_inputs_for_generation(model)
            model = PeftModel.from_pretrained(
                model, args.checkpoint_path, torch_dtype=torch.bfloat16
            ).to(local_rank)

        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

    patch_sdar_for_eval(model, block_length=args.block_length)
    patch_llada2_block_causal_attention(model, args.block_length)
    model = model.to(torch.bfloat16)

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if args.checkpoint_path:
        model_name = args.checkpoint_path.split("/")
        if not model_name[-1]:
            model_name = model_name[:-1]
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name += f"_fs{args.few_shot}"
    if args.suffix:
        model_name += f"_{args.suffix}"

    # Append r1 suffix to distinguish from fixed-block eval
    model_name += "_r1"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = (
        f"{args.output_dir}/{model_name}_{args.dataset}"
        f"_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    )
    if local_rank == 0:
        print(f"Saving R1 generations to {filename}")

    with open(filename, "w") as f:
        f.write('{\n  "generations": [\n')
        first = [True]

        def real_time_callback(sample):
            if not first[0]:
                f.write(",\n")
            else:
                first[0] = False
            lines = json.dumps(sample, indent=2, ensure_ascii=True).splitlines()
            for line in lines:
                f.write("    " + line + "\n")
            f.flush()
            torch.cuda.empty_cache()

        metrics = r1_evaluate(
            model, tokenizer, dataloader,
            controller=controller,
            gen_length=args.gen_length,
            fallback_block_length=args.block_length,
            steps=args.diffusion_steps,
            real_time_callback=real_time_callback,
            dataset=args.dataset,
        )

        f.write("\n  ],\n")
        f.write('  "metrics": {\n')
        f.write(f'    "wall_time": {metrics["wall_time"]},\n')
        f.write(f'    "total_processed": {metrics["total_processed"]}\n')
        f.write("  },\n")
        f.write(f'  "model_path": "{args.model_path}",\n')
        f.write(f'  "checkpoint_path": "{args.checkpoint_path}",\n')
        f.write(f'  "gen_length": {args.gen_length},\n')
        f.write(f'  "diffusion_steps": {args.diffusion_steps},\n')
        f.write(f'  "block_length_fallback": {args.block_length},\n')
        f.write(f'  "r1_block_size_used": {metrics.get("r1_block_size_used", args.block_length)},\n')
        f.write(f'  "r1_controller_path": "{args.r1_controller_path}"\n')
        f.write("}\n")

    cleanup_ddp()
