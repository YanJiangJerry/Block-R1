# Only be called during testing evaluation by bash eval_all.sh
# By default using b1 prompts
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Compat shim: older datasets lib lacks 'List' feature type that newer cached
# metadata (e.g. MMLU-Pro, MMLU) may reference. Register Sequence as alias.
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
import torch.distributed as dist

from rl.trainers.dynamic_generate import dynamic_generate
from rl.eval.generate import generate

# Import datasets
from countdown import CTDDataset
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from sudoku import SudokuDataset
from mbpp import MBPPDataset
from humaneval import HumanEvalDataset
from kodcode import KodCodeDataset
from mmlu import MMLUDataset
from mmlu_pro import MMLUProDataset
from hellaswag import HellaSwagDataset
from arc_c import ARCCDataset
from arc_e import ARCEDataset
from gpqa import GPQADataset
from knights_and_knaves import KnightsAndKnavesDataset

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "mbpp": MBPPDataset,
    "humaneval": HumanEvalDataset,
    "kodcode": KodCodeDataset,  # Use last 500 samples of KodCode for testing
    "mmlu": MMLUDataset,
    "mmlu_pro": MMLUProDataset,
    "hellaswag": HellaSwagDataset,
    "arc_c": ARCCDataset,
    "arc_e": ARCEDataset,
    "gpqa": GPQADataset,
    "knights_and_knaves": KnightsAndKnavesDataset,
}

# Import LoraConfig to manually instantiate config
from peft import PeftModel, PeftConfig, LoraConfig

from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from rl.llada2_compat import (
    ensure_transformers_kwargs,
    load_diffusion_model,
    patch_llada2_block_causal_attention,
    patch_prepare_inputs_for_generation,
    patch_sdar_for_eval,
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


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=256,
    temperature=0.0,
    cfg_scale=0.0,
    steps=128,
    block_length=32,
    real_time_callback=None,  # New callback for real-time actions
    dataset=None,  # Target dataset name for any dataset-specific handling (e.g. mbpp per-sample generation
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        # out, _ = dynamic_generate(
        #     model=model,
        #     prompt=input_ids,
        #     steps=steps,
        #     gen_length=gen_length,
        #     temperature=temperature,
        #     cfg_scale=cfg_scale,
        #     remasking="low_confidence",
        #     tokenizer=tokenizer,
        # )
        # For mbpp dataset, generate small batch sample to avoid OOM when batching large code outputs.
        if dataset == "mbpp" and "test_list" in batch:
            sb = 16  # micro-batch size，
            generated_texts = []
            N = input_ids.size(0)
            for start in range(0, N, sb):
                end = min(start + sb, N)
                mini_inputs = input_ids[start:end]
                out_mini = generate(
                    model,
                    mini_inputs,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
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
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
            )

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=False
            )

        # Build example results
        example_result = []
        for j in range(len(gt_answers)):
            result = {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            # For code generation tasks with test cases (e.g. mbpp), also save the test cases in the results for later evaluation.
            if "test_list" in batch:
                result["test_list"] = batch["test_list"][j]
            example_result.append(result)

        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

        # Real-time callback action
        if real_time_callback is not None:
            for sample in example_result:
                real_time_callback(sample)

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

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
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "gsm8k",
            "math",
            "countdown",
            "sudoku",
            "game24",
            "mbpp",
            "humaneval",
            "kodcode",
            "mmlu",
            "mmlu_pro",
            "hellaswag",
            "arc_c",
            "arc_e",
            "gpqa",
            "knights_and_knaves",
        ],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    args = parser.parse_args()

    # By default set diffusion steps to half of gen length
    args.diffusion_steps = args.gen_length // 2
    # Countdown and Sudoku by default use 256 testing samples generated by d1 papers
    num_evals = {
        "gsm8k": -1,
        "math": -1,
        "countdown": -1,
        "mbpp": -1,
        "humaneval": -1,
        "sudoku": 256,
        "kodcode": -1,
        "mmlu": -1,
        "mmlu_pro": -1,
        "hellaswag": -1,
        "arc_c": -1,
        "arc_e": -1,
        "gpqa": -1,
        "knights_and_knaves": -1,
    }

    ensure_transformers_kwargs()
    device = torch.device(f"cuda:{local_rank}")

    # Keep model code stable across time: if the caller doesn't pin a revision,
    # default to known-good commits (same policy as rl/run_train.py).
    _PINNED_REVISIONS = {
        "inclusionAI/LLaDA2.1-mini": "bbb5715c881500b34234071e68dbf38c3d657c4e",
        "inclusionAI/LLaDA2.0-mini": "d23215abc5f5675daf171f6739d0386eab53f712",
    }
    revision = str(args.revision).strip() or _PINNED_REVISIONS.get(str(args.model_path))

    model = load_diffusion_model(
        args.model_path,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, revision=revision
    )

    if args.checkpoint_path:
        # Universal Auto-Sanitization
        try:
            import inspect
            from peft import LoraConfig

            # 1. Load the JSON manually
            config_path = os.path.join(args.checkpoint_path, "adapter_config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # 2. Introspect LoraConfig to find out what arguments it ACTUALLY accepts
            # Get the signature of the __init__ method
            signature = inspect.signature(LoraConfig.__init__)
            # Get the list of valid parameter names
            valid_params = set(signature.parameters.keys())

            # 3. Create a clean dictionary containing ONLY valid parameters
            # This automatically drops 'corda_config', 'eva_config', 'use_qalora', etc.
            sanitized_config_dict = {
                k: v for k, v in config_dict.items() if k in valid_params
            }

            # Optional: Print what was removed for debugging
            removed_keys = set(config_dict.keys()) - set(sanitized_config_dict.keys())
            if len(removed_keys) > 0 and local_rank == 0:
                print(
                    f"[Warning] Auto-removed incompatible config keys: {removed_keys}"
                )

            # 4. Instantiate Config and Model
            config = LoraConfig(**sanitized_config_dict)

            # DreamModel is a diffusion model and does not have prepare_inputs_for_generation.
            # PEFT's PeftModelForCausalLM.__init__ unconditionally accesses this attribute,
            # so we patch it onto the instance before loading to avoid AttributeError.
            # This does NOT affect Dream's own generation logic.
            patch_prepare_inputs_for_generation(model)

            model = PeftModel.from_pretrained(
                model, args.checkpoint_path, config=config, torch_dtype=torch.bfloat16
            ).to(local_rank)

        except Exception as e:
            # Fallback
            print(f"Rank {local_rank}: Auto-fix failed ({e}), trying standard load...")
            # Same patch for the fallback path
            patch_prepare_inputs_for_generation(model)
            model = PeftModel.from_pretrained(
                model, args.checkpoint_path, torch_dtype=torch.bfloat16
            ).to(local_rank)

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if local_rank == 0:
                print(f"Rank {local_rank}: Parameters synchronized")

    patch_sdar_for_eval(model, block_length=args.block_length)
    patch_llada2_block_causal_attention(model, args.block_length)
    model = model.to(torch.bfloat16)

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill the begin of <reasoning> tag for all models
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        if not model_name[-1]:
            model_name = model_name[:-1]
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{model_name}_{args.dataset}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")

    # Default temperature 0.0 for eval
    with open(filename, "w") as f:
        f.write('{\n  "generations": [\n')
        first = [True]

        def real_time_callback(sample):
            if not first[0]:
                f.write(",\n")
            else:
                first[0] = False

            # ensure_ascii=True is required: ensure_ascii=False leaves U+2028/U+2029
            # (Unicode line separators) unescaped, and .splitlines() treats them as
            # line breaks, corrupting the JSON string values.
            lines = json.dumps(sample, indent=2, ensure_ascii=True).splitlines()
            for line in lines:
                f.write("    " + line + "\n")
            f.flush()
            torch.cuda.empty_cache()

        metrics = evaluate(
            model,
            tokenizer,
            dataloader,
            gen_length=args.gen_length,
            block_length=args.block_length,
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
        f.write(f'  "revision": "{revision}",\n')
        f.write(f'  "checkpoint_path": "{args.checkpoint_path}",\n')
        f.write(f'  "gen_length": {args.gen_length},\n')
        f.write(f'  "diffusion_steps": {args.diffusion_steps},\n')
        f.write(f'  "block_length": {args.block_length}\n')
        f.write("}\n")

    cleanup_ddp()
