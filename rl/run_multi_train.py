"""
R1 Multi-Domain Training Entry Point for dLLM-R1.

Standalone entry for R1 (dynamic block-size bandit + per-domain rewards).
Requires ``trainer_type`` like ``r1_wd1`` / ``r1_d1`` / ``r1_stable_drl`` and ``--use_r1 true``.
Guru (LLM360/guru-RL-92k): ``--r1_domains guru`` → ``load_guru_rl_train`` (rows carry
``domain`` = ``guru_math`` / ``guru_code`` / … for R1 centroids + Q) +
``guru_unified_reward_func`` → ``rl.eval.guru.default_compute_score``.

Does not use the single-domain ``rl/run_train.py`` pipeline.

Usage:
  accelerate launch rl/run_multi_train.py \
    --config rl/train.yaml \
    --model_path inclusionAI/LLaDA2.1-mini \
    --trainer_type r1_wd1 \
    --use_r1 true \
    --r1_domains "math,countdown,kodcode" \
    --r1_block_size_candidates "16,32,64,128" \
    ...

  # Guru-only (see reproduce/wd1/r1_wd1_guru.sh):
  #   --use_r1 true --r1_domains guru --trainer_type r1_wd1
"""

import os
import warnings
import logging
import inspect

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def _noop(*_args, **_kwargs):
    return


warnings.showwarning = _noop
warnings.warn = _noop
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for _name in ["transformers", "torch", "pydantic", "torch.distributed.run", "accelerate"]:
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass

import torch
import torch.distributed as dist
import wandb
from datasets import Dataset, concatenate_datasets

try:
    from data_utils import (  # type: ignore
        get_countdown_questions,
        get_gsm8k_questions,
        get_math_questions,
        get_sudoku_questions,
        get_mbpp_questions,
        get_humaneval_questions,
        get_kodcode_light_rl_10k,
        get_mmlu_questions,
        get_mmlu_pro_questions,
        get_hellaswag_questions,
        get_arc_c_questions,
        get_arc_e_questions,
        get_gpqa_questions,
        get_knights_and_knaves_questions,
        set_random_seed,
        set_trainer_type,
    )
except ModuleNotFoundError:
    from rl.data_utils import (
        get_countdown_questions,
        get_gsm8k_questions,
        get_math_questions,
        get_sudoku_questions,
        get_mbpp_questions,
        get_humaneval_questions,
        get_kodcode_light_rl_10k,
        get_mmlu_questions,
        get_mmlu_pro_questions,
        get_hellaswag_questions,
        get_arc_c_questions,
        get_arc_e_questions,
        get_gpqa_questions,
        get_knights_and_knaves_questions,
        set_random_seed,
        set_trainer_type,
    )
from peft import LoraConfig

_orig_lora_init = LoraConfig.__init__


def _robust_lora_init(self, *args, **kwargs):
    valid_params = set(inspect.signature(_orig_lora_init).parameters.keys())
    clean_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    _orig_lora_init(self, *args, **clean_kwargs)


LoraConfig.__init__ = _robust_lora_init

import transformers.utils as _transformers_utils

if not hasattr(_transformers_utils, "TransformersKwargs"):
    from typing import TypedDict

    class TransformersKwargs(TypedDict):
        pass

    _transformers_utils.TransformersKwargs = TransformersKwargs

try:
    from reward_func import (  # type: ignore
        boxed_and_answer_tags_format_reward,
        correctness_reward_func,
        correctness_reward_func_math,
        countdown_reward_func,
        int_reward_func,
        knights_knaves_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        sudoku_reward_func,
        xmlcount_reward_func,
        block_format_reward,
        code_reward_func,
        get_code_format_reward,
        code_reward,
        mc_reward_func,
        guru_unified_reward_func,
        _sanitize_guru_extra_info,
        _fill_guru_question_in_extra,
    )
except ModuleNotFoundError:
    from rl.reward_func import (
        boxed_and_answer_tags_format_reward,
        correctness_reward_func,
        correctness_reward_func_math,
        countdown_reward_func,
        int_reward_func,
        knights_knaves_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        sudoku_reward_func,
        xmlcount_reward_func,
        block_format_reward,
        code_reward_func,
        get_code_format_reward,
        code_reward,
        mc_reward_func,
        guru_unified_reward_func,
        _sanitize_guru_extra_info,
        _fill_guru_question_in_extra,
    )
from rl.eval.guru_dataset import GURU_R1_DOMAIN_KEYS, load_guru_rl_train

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
)
from trl import ModelConfig, TrlParser

from rl.trainers.diffu_grpo_config import DiffuGRPOConfig
from rl.trainers.eval_callback import AccuracyEvalCallback, generate as eval_generate
from rl.trainers.cross_domain_generate import (
    BlockSizeController,
    create_multi_domain_reward_func,
    compute_prompt_embedding,
    get_r1_trainer_class,
)
from rl.trainers.r1_adaptive_proto import AdaptiveProtoBlockController

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


import re
import pandas as pd
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback

from rl.eval.parse_and_get_acc import (
    parse_gsm_answers,
    parse_math_answers,
    parse_countdown_answers,
    parse_sudoku_answers,
    parse_code_answers,
    parse_mc_answers,
    parse_knights_knaves_answers,
)


class R1ControllerSaveCallback(TrainerCallback):
    """Save r1.json alongside every checkpoint so eval can pick any training step."""

    def __init__(self, controller):
        self.controller = controller

    def on_save(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        if self.controller is None:
            return
        import json as _json
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save inside the checkpoint dir (step-specific)
        with open(os.path.join(ckpt_dir, "r1.json"), "w") as f:
            _json.dump(self.controller.state_dict(), f, indent=2)
        # Also save to output_dir root (always the latest)
        with open(os.path.join(args.output_dir, "r1.json"), "w") as f:
            _json.dump(self.controller.state_dict(), f, indent=2)
        print(f"[R1] Saved r1.json at step {state.global_step}")


class R1PerDomainEvalCallback(TrainerCallback):
    """
    Per-domain eval callback for R1 multi-domain training.

    Key differences from the base AccuracyEvalCallback:
      1. Block size: if ``domain_name`` is in ``controller.domains``, use that domain's Q;
         otherwise (e.g. Guru train + gsm8k/math eval callbacks) embed the eval batch and
         ``select_block_size_by_embedding`` so the nearest *training* centroid picks Q
         (same zero-shot idea as ``rl.eval.r1_eval``). DSCB still uses the proto branch.
      2. Logs to eval/accuracy/{domain_name} on wandb
      3. Logs eval/block_size/{domain_name} for the last batch's selected block size
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        domain_name,
        block_size_controller,
        gen_length=256,
        steps=128,
        fallback_block_length=32,
        batch_size=4,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.domain_name = domain_name
        self.controller = block_size_controller
        self.gen_length = gen_length
        self.steps = steps
        self.fallback_block_length = fallback_block_length
        self.dataloader = DataLoader(
            eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate_fn,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        accelerator = kwargs["accelerator"]
        model = kwargs["model"]

        # R1: domain-oracle Q (classic) vs nearest-prototype + Q (adaptive DSCB)
        block_length = self.fallback_block_length
        use_proto_routing = (
            self.controller is not None
            and hasattr(self.controller, "observe_prompt_embeddings")
        )
        # Guru-only (or any train/eval domain mismatch): eval benchmarks are not in
        # controller.domains — route by nearest training-domain centroid + Q (same idea as r1_eval).
        use_centroid_eval_routing = (
            self.controller is not None
            and not use_proto_routing
            and self.domain_name not in self.controller.domains
        )
        last_block_logged = self.fallback_block_length
        if not use_proto_routing and self.controller is not None:
            if self.domain_name in self.controller.domains:
                last_block_logged = int(
                    self.controller.get_best_block_size(self.domain_name)
                )
            else:
                last_block_logged = int(self.fallback_block_length)

        eval_dataloader = accelerator.prepare(self.dataloader)
        all_generations = []
        if accelerator.is_main_process:
            eval_dataloader = tqdm(
                eval_dataloader, desc=f"Eval {self.domain_name} (bs={block_length})", leave=True,
            )

        for batch in eval_dataloader:
            gt_answers = batch["answers"]
            questions = batch["questions"]
            prompts = batch["prompts"]
            test_lists = batch.get("test_list", [None] * len(gt_answers))

            # Match R1 training tokenization for embedding routing (DSCB / centroid).
            if (
                (use_proto_routing or use_centroid_eval_routing)
                and self.controller is not None
                and batch.get("prompts") is not None
            ):
                enc = self.tokenizer(
                    text=list(batch["prompts"]),
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"]
            else:
                input_ids = batch["input_ids"]

            # Retokenization leaves tensors on CPU; prepared batches may already be on device.
            # eval_callback.generate builds x on prompt.device — must match model device.
            input_ids = input_ids.to(accelerator.device)

            batch_block_length = block_length
            if use_proto_routing and self.controller is not None:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                try:
                    emb = compute_prompt_embedding(model, input_ids.to(model.device), pad_id)
                    if emb is not None:
                        batch_centroid = emb.mean(dim=0)
                        batch_block_length = int(
                            self.controller.select_block_size_by_embedding(batch_centroid)
                        )
                except Exception:
                    batch_block_length = self.fallback_block_length
            elif use_centroid_eval_routing:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                try:
                    emb = compute_prompt_embedding(model, input_ids.to(model.device), pad_id)
                    if emb is not None:
                        batch_centroid = emb.mean(dim=0)
                        batch_block_length = int(
                            self.controller.select_block_size_by_embedding(batch_centroid)
                        )
                    else:
                        batch_block_length = int(self.fallback_block_length)
                except Exception:
                    batch_block_length = int(self.fallback_block_length)
            elif self.controller is not None:
                batch_block_length = int(
                    self.controller.get_best_block_size(self.domain_name)
                )

            try:
                from rl.llada2_compat import is_llada2_moe, set_llada2_eval_block_length

                if is_llada2_moe(model):
                    set_llada2_eval_block_length(model, batch_block_length)
            except Exception:
                pass

            with torch.no_grad():
                out = eval_generate(
                    model, input_ids, self.tokenizer,
                    steps=self.steps, gen_length=self.gen_length,
                    block_length=batch_block_length, temperature=0.0, cfg_scale=0.0,
                    remasking="low_confidence",
                    disable_bar=not accelerator.is_main_process,
                )

            generated_texts = self.tokenizer.batch_decode(
                out[:, -self.gen_length:], skip_special_tokens=False,
            )
            for j in range(len(gt_answers)):
                result = {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                    "test_list": test_lists[j] if test_lists[j] is not None else "",
                }
                all_generations.append(result)

            last_block_logged = batch_block_length

        all_generations = gather_object(all_generations)

        if accelerator.is_main_process:
            true_n = len(self.eval_dataset) if self.eval_dataset is not None else 0
            all_generations = _dedupe_and_truncate_generations(all_generations, true_n)
            dataset_cls_name = getattr(self.eval_dataset.__class__, "__name__", "").lower()
            json_data = {"generations": all_generations}
            accuracy = 0.0
            num_correct = None
            num_total = None

            try:
                if "gsm" in dataset_cls_name:
                    c, p, _, _ = parse_gsm_answers(json_data=json_data)
                    accuracy = c / p if p > 0 else 0.0
                    num_correct, num_total = c, p
                elif "math" in dataset_cls_name:
                    c, p, _, _ = parse_math_answers(json_data=json_data)
                    accuracy = c / p if p > 0 else 0.0
                    num_correct, num_total = c, p
                elif "countdown" in dataset_cls_name or "ctd" in dataset_cls_name:
                    c, p, _, _ = parse_countdown_answers(json_data=json_data)
                    accuracy = c / p if p > 0 else 0.0
                    num_correct, num_total = c, p
                elif "sudoku" in dataset_cls_name:
                    c, t, _, _ = parse_sudoku_answers(json_data=json_data)
                    accuracy = c / t if t > 0 else 0.0
                    num_correct, num_total = c, t
                elif "mbpp" in dataset_cls_name or "humaneval" in dataset_cls_name or "kodcode" in dataset_cls_name:
                    c, p, _, _ = parse_code_answers(json_data=json_data)
                    accuracy = c / p if p > 0 else 0.0
                    num_correct, num_total = c, p
                elif any(k in dataset_cls_name for k in ["mmlu", "hellaswag", "arc", "gpqa"]):
                    c, p, _, _ = parse_mc_answers(json_data=json_data)
                    accuracy = c / p if p > 0 else 0.0
                    num_correct, num_total = c, p
                elif "knights" in dataset_cls_name:
                    c, p, _, _ = parse_knights_knaves_answers(json_data=json_data)
                    accuracy = c / p if p > 0 else 0.0
                    num_correct, num_total = c, p
            except Exception as e:
                print(f"[R1 Eval] Parsing error for {self.domain_name}: {e}")

            routing_tag = (
                "proto"
                if use_proto_routing
                else ("centroid" if use_centroid_eval_routing else "domain_q")
            )
            if num_correct is not None and num_total is not None:
                print(
                    f"[R1 Eval] {self.domain_name}: Eval Accuracy: {accuracy:.4f} "
                    f"({int(num_correct)}/{int(num_total)})"
                )
            else:
                print(
                    f"[R1 Eval] {self.domain_name}: Eval Accuracy: {accuracy:.4f} "
                    f"({len(all_generations)}/{len(all_generations)})"
                )

            if args.report_to and "wandb" in args.report_to:
                wandb.log({
                    f"eval/accuracy/{self.domain_name}": accuracy,
                })

        accelerator.wait_for_everyone()


def _r1_per_domain_eval_callback(domain_name: str, **kwargs):
    """
    Hugging Face ``CallbackHandler`` warns when multiple callbacks share the same class.
    Guru training registers one eval callback per benchmark; use a unique subclass per
    ``domain_name`` so the list stays valid without noisy duplicates.
    """
    safe = "".join(c if c.isalnum() else "_" for c in str(domain_name)) or "default"
    cls = type(f"R1PerDomainEval_{safe}", (R1PerDomainEvalCallback,), {})
    # Pass domain_name explicitly — do not also pass domain_name= in **kwargs (duplicate arg).
    kwargs.pop("domain_name", None)
    return cls(domain_name=domain_name, **kwargs)


def _dedupe_and_truncate_generations(all_generations: list[dict], target_n: int) -> list[dict]:
    """
    R1 eval uses `accelerator.prepare(dataloader)` which can repeat samples to keep even batches across ranks.
    For accuracy denominators to match the true eval set size, dedupe + truncate after gather.
    """
    if target_n <= 0 or not all_generations:
        return all_generations
    seen = set()
    uniq: list[dict] = []
    for item in all_generations:
        q = item.get("question", "")
        gt = item.get("ground_truth", "")
        key = (q, str(gt))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
        if len(uniq) >= target_n:
            break
    if len(uniq) < min(target_n, len(all_generations)):
        return all_generations[:target_n]
    return uniq


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

# When training on `guru`, run periodic eval on these benchmarks (in addition to any
# overlapping domains already in r1_domains). See R1PerDomainEvalCallback.
GURU_TRAIN_EVAL_DOMAINS = (
    "gsm8k",
    "math",
    "knights_and_knaves",
    "sudoku",
    "countdown",
    "humaneval",
    "mbpp",
)
SUB_SAMPLE_MAP = {
    "gsm8k": -1,
    "math": -1,
    "countdown": -1,
    "sudoku": -1,
    "mbpp": -1,
    "humaneval": -1,
    "kodcode": -1,
    "knights_and_knaves": -1,
    "mmlu": 500,
    "mmlu_pro": 500,
    "hellaswag": 500,
    "arc_c": 500,
    "arc_e": 500,
    "gpqa": 500,
}

# Domain -> dataset loader (train split)
DOMAIN_TRAIN_LOADER = {
    "gsm8k": lambda: get_gsm8k_questions("train"),
    "math": lambda: get_math_questions("train"),
    "countdown": lambda: get_countdown_questions("train"),
    "sudoku": lambda: get_sudoku_questions(),
    "mbpp": lambda: get_kodcode_light_rl_10k("train"),
    "humaneval": lambda: get_kodcode_light_rl_10k("train"),
    "kodcode": lambda: get_kodcode_light_rl_10k("train"),
    "mmlu": lambda: get_mmlu_questions("auxiliary_train"),
    "mmlu_pro": lambda: get_mmlu_questions("auxiliary_train"),
    "hellaswag": lambda: get_hellaswag_questions("train"),
    "arc_c": lambda: get_arc_c_questions("train"),
    "arc_e": lambda: get_arc_e_questions("train"),
    "gpqa": lambda: get_mmlu_questions("auxiliary_train"),
    "knights_and_knaves": lambda: get_knights_and_knaves_questions("train"),
    # Guru (HF): R1 ``domain`` is per coarse skill bucket (guru_math, guru_code, …); reward
    # still uses guru_data_source → rl.eval.guru. Eval benchmarks (gsm8k, …) are not those
    # names — R1PerDomainEvalCallback uses prompt embedding → nearest training centroid + Q.
    "guru": lambda: load_guru_rl_train(
        max_samples=int(os.environ["GURU_MAX_SAMPLES"])
        if os.environ.get("GURU_MAX_SAMPLES", "").isdigit()
        else None
    ),
}

# Domain -> reward functions
DOMAIN_REWARD_MAP = {
    "gsm8k": [xmlcount_reward_func, int_reward_func, correctness_reward_func],
    "math": [correctness_reward_func_math, boxed_and_answer_tags_format_reward],
    "countdown": [countdown_reward_func],
    "sudoku": [sudoku_reward_func],
    "mbpp": [get_code_format_reward(language="python"), code_reward],
    "humaneval": [get_code_format_reward(language="python"), code_reward],
    "kodcode": [get_code_format_reward(language="python"), code_reward],
    "mmlu": [mc_reward_func],
    "mmlu_pro": [mc_reward_func],
    "hellaswag": [mc_reward_func],
    "arc_c": [mc_reward_func],
    "arc_e": [mc_reward_func],
    "gpqa": [mc_reward_func],
    "knights_and_knaves": [knights_knaves_reward_func],
    "guru": [guru_unified_reward_func],
}


def expand_r1_controller_domains(names):
    """Replace loader name ``guru`` with per-bucket R1 domains (centroids + Q per sub-domain)."""
    out = []
    for n in names:
        if n == "guru":
            out.extend(GURU_R1_DOMAIN_KEYS)
        else:
            out.append(n)
    return out


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _print_guru_reward_diagnostics(train_set) -> None:
    """
    One-shot stats: flat Guru rewards (all 0 in a GRPO group) → zero_std_ratio=1 →
    advantages≈0. Then: r1_wd1 PSR/NSR cancel (loss≈0, grad≈0); r1_d1 clipped
    policy term is 0 but β·KL remains (loss may still print 0.0 if KL is tiny).
    Common causes: missing \\boxed{...}; wrong answers; codegen/table failures.
    """
    cols = getattr(train_set, "column_names", None)
    if not cols or "guru_data_source" not in cols:
        return
    import random as _random

    n = len(train_set)
    k = min(8000, n)
    rng = _random.Random(42)
    idxs = rng.sample(range(n), k) if k < n else list(range(n))
    sub = train_set.select(idxs)
    sources = sub["guru_data_source"]
    stem_web_n = sum(1 for x in sources if str(x or "").startswith("stem_web"))
    math_n = sum(1 for x in sources if str(x or "").startswith("math"))
    codegen_n = sum(1 for x in sources if str(x or "").startswith("codegen"))
    table_n = sum(1 for x in sources if str(x or "").startswith("table"))
    print(
        f"[R1] Guru data_source mix (sample {k}/{n} rows): "
        f"stem_web*={stem_web_n} ({100 * stem_web_n / max(k, 1):.1f}%, local naive_dapo), "
        f"math*={math_n}, codegen*={codegen_n}, table*={table_n}"
    )
    print(
        "[R1] Tip: many scorers need \\boxed{...} in model output; early diffusion "
        "samples often score 0 until the policy learns format."
    )


def _guru_reward_oracle_smoke_test(train_set) -> None:
    """
    Call the same ``guru_unified_reward_func`` as training with a fake completion
    ``\\boxed{ground_truth}``. If oracle_reward≈1 for math* rows but train reward≈0,
    the scorer is OK and the policy output is missing \\boxed / wrong / codegen fail.
    """
    cols = getattr(train_set, "column_names", None)
    if not cols or "guru_data_source" not in cols:
        return
    import random as _random

    rng = _random.Random(44)
    n = len(train_set)
    k = min(12, n)
    idxs = rng.sample(range(n), k)
    print(
        "[R1] Guru oracle smoke (same reward as training, completion=\\boxed{GT first line}):"
    )
    math_ok = 0
    math_n = 0
    for j, idx in enumerate(idxs):
        row = train_set[idx]
        ds = str(row.get("guru_data_source") or "").strip()
        gt_raw = row.get("guru_reward_ground_truth")
        if isinstance(gt_raw, bytes):
            gt_s = gt_raw.decode("utf-8", errors="replace")
        else:
            gt_s = str(gt_raw or "")
        ex_raw = row.get("guru_extra_info")
        extra = _sanitize_guru_extra_info(ex_raw) if ex_raw is not None else {}
        prompt = row.get("prompt")
        rm = extra.get("reward_metric") if isinstance(extra, dict) else None
        if ds.startswith("stem_web") or rm == "math_llm_judge":
            extra = _fill_guru_question_in_extra(extra, prompt)

        if not ds:
            print(f"  #{j} idx={idx} empty guru_data_source")
            continue
        if not gt_s.strip():
            print(
                f"  #{j} idx={idx} ds={ds[:44]}... empty guru_reward_ground_truth "
                "(cannot grade → training reward 0 for these rows)"
            )
            continue
        inner = gt_s.split("\n")[0].strip()[:500]
        solution = f"\\boxed{{{inner}}}"
        comp = [{"role": "assistant", "content": solution}]

        try:
            r = guru_unified_reward_func(
                prompts=[prompt],
                completions=[comp],
                guru_data_source=[ds],
                guru_reward_ground_truth=[gt_s],
                guru_extra_info=[extra],
            )
            sc = float(r[0]) if r else 0.0
        except Exception as e:
            print(f"  #{j} idx={idx} ds={ds[:44]}... oracle EXC {e!r}")
            continue

        print(f"  #{j} idx={idx} ds={ds[:52]}... oracle_reward={sc:.4f}")
        if ds.startswith("math") or ds.startswith("stem_web"):
            math_n += 1
            if sc >= 0.5:
                math_ok += 1

    if math_n > 0:
        print(
            f"[R1] Oracle summary: math* + stem_web* (local naive_dapo) {math_ok}/{math_n} "
            f"with score>=0.5. If this is high but train reward=0 → model outputs "
            f"rarely match \\boxed{{...}} / ground_truth (not a WD1 bug)."
        )


def create_multi_domain_dataset(domain_names, seed=42, balance_domains: bool = True):
    """
    Load and merge training datasets from multiple domains.

    Each sample gets a 'domain' field for reward routing and block size selection.
    HuggingFace's concatenate_datasets automatically fills missing columns with None,
    so datasets with different schemas are safely merged.
    """
    all_datasets = []
    per_domain_sizes = {}

    for domain_name in domain_names:
        if domain_name not in DOMAIN_TRAIN_LOADER:
            raise ValueError(
                f"Unknown domain: {domain_name}. "
                f"Available: {list(DOMAIN_TRAIN_LOADER.keys())}"
            )

        if is_main_process():
            print(f"[R1] Loading training data for domain: {domain_name}")

        dataset = DOMAIN_TRAIN_LOADER[domain_name]()
        dataset = dataset.shuffle(seed=seed)

        if domain_name in ["countdown", "sudoku"]:
            dataset = dataset.select(range(0, max(len(dataset) - 500, 1)))
        elif domain_name == "mbpp":
            if len(dataset) >= 974:
                dataset = dataset.select(range(500, 974))
        # Guru (HF) — keep full curated split unless GURU_MAX_SAMPLES is set in loader

        # Tag each sample with its R1 domain (Guru rows already have guru_* from the loader)
        if domain_name != "guru":
            dataset = dataset.map(lambda _x, dn=domain_name: {"domain": dn})

        per_domain_sizes[domain_name] = len(dataset)
        if is_main_process():
            print(f"[R1]   {domain_name}: {len(dataset)} training samples")

        all_datasets.append(dataset)

    # Balance domains so each contributes equally to training.
    # Without this, large domains can dominate sampling and make Q-values for
    # smaller domains change slowly (or appear stuck).
    if balance_domains and len(all_datasets) > 1:
        target = min(per_domain_sizes.values())
        if is_main_process():
            print(f"[R1] Balancing domains: sampling {target} per domain (min size)")
        balanced = []
        for dname, ds in zip(domain_names, all_datasets):
            # Datasets are already shuffled with the same seed above, so taking the
            # first `target` examples is deterministic and random w.r.t. original order.
            balanced.append(ds.select(range(target)))
        all_datasets = balanced

    # concatenate_datasets auto-fills missing columns with None across schemas
    merged = concatenate_datasets(all_datasets)
    merged = merged.shuffle(seed=seed)

    if is_main_process():
        print(f"[R1] Total merged training samples: {len(merged)}")
        print(f"[R1] Unified columns: {merged.column_names}")

    return merged


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    # This entrypoint is R1-only (dynamic block-size + multi-domain rewards).
    tt = (grpo_config.trainer_type or "").strip()
    if not tt.startswith("r1_"):
        raise ValueError(
            f"run_multi_train.py expects trainer_type r1_* (e.g. r1_wd1, r1_d1, r1_stable_drl). Got {tt!r}. "
            "Use rl/run_train.py for non-R1 training."
        )
    if not getattr(grpo_config, "use_r1", False):
        raise ValueError(
            "run_multi_train.py requires --use_r1 true (see reproduce/wd1/r1_wd1_guru.sh)."
        )

    # Parse R1 domains
    if not grpo_config.r1_domains:
        raise ValueError(
            "R1 requires --r1_domains to be set. "
            "E.g. --r1_domains 'math,countdown,kodcode' or --r1_domains guru"
        )
    domain_names = [d.strip() for d in grpo_config.r1_domains.split(",")]
    if is_main_process():
        print(f"[R1] Domains: {domain_names}")
        print(f"[R1] Trainer type: {grpo_config.trainer_type}")

    # R1 doesn't use B1's \block markers, so use baseline prompt templates.
    # Extract the underlying RL algorithm from the trainer_type (e.g. r1_wd1 -> wd1)
    base_algo = grpo_config.trainer_type.replace("r1_", "")
    set_trainer_type(base_algo)

    # Create multi-domain dataset
    train_set = create_multi_domain_dataset(domain_names, seed=grpo_config.seed, balance_domains=True)
    if len(train_set) == 0:
        raise ValueError(
            "[R1] Training dataset is empty (check domain loaders, GURU_MAX_SAMPLES, or guru rows dropped for empty data_source)."
        )

    if "guru" in domain_names and is_main_process():
        _print_guru_reward_diagnostics(train_set)
        _guru_reward_oracle_smoke_test(train_set)

    # GRPO / TRL: global prompt batch must split evenly into num_generations groups.
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    pdb = int(grpo_config.per_device_train_batch_size)
    gas = int(getattr(grpo_config, "gradient_accumulation_steps", 1) or 1)
    global_bs = pdb * max(ws, 1) * max(gas, 1)
    ng = int(getattr(grpo_config, "num_generations", 1) or 1)
    if ng > 0 and global_bs % ng != 0:
        raise ValueError(
            "[R1] GRPO needs (per_device_train_batch_size * WORLD_SIZE * gradient_accumulation_steps) "
            f"divisible by num_generations; got {pdb}*{ws}*{gas}={global_bs}, num_generations={ng}."
        )

    if (
        getattr(grpo_config, "dataloader_drop_last", False)
        and len(train_set) < global_bs
        and is_main_process()
    ):
        warnings.warn(
            f"[R1] len(train_set)={len(train_set)} < global train batch {global_bs} "
            f"and dataloader_drop_last=True — a rank may see no batches. "
            f"Use more data, fewer GPUs, smaller per_device_train_batch_size, or dataloader_drop_last=false.",
            stacklevel=1,
        )

    # Create unified reward function
    active_reward_map = {d: DOMAIN_REWARD_MAP[d] for d in domain_names}
    unified_reward_func = create_multi_domain_reward_func(active_reward_map)
    reward_functions = [unified_reward_func]

    # Parse block size candidates
    block_size_candidates = [
        int(b.strip()) for b in grpo_config.r1_block_size_candidates.split(",")
    ]

    # Create block size controller (fixed per-domain centroids vs DSCB adaptive prototypes)
    if getattr(grpo_config, "r1_adaptive_proto", False):
        controller = AdaptiveProtoBlockController(
            max_prototypes=int(grpo_config.r1_max_prototypes),
            block_size_candidates=block_size_candidates,
            gen_length=grpo_config.max_completion_length,
            lr=grpo_config.r1_block_size_lr,
            default_block_size=grpo_config.block_length,
            gamma_confidence=float(grpo_config.r1_proto_gamma),
            beta_sim=float(grpo_config.r1_proto_beta_sim),
            min_samples_stat=int(grpo_config.r1_proto_min_samples_stat),
            exploration_rate=grpo_config.r1_exploration_rate,
        )
    else:
        controller = BlockSizeController(
            domains=expand_r1_controller_domains(domain_names),
            block_size_candidates=block_size_candidates,
            gen_length=grpo_config.max_completion_length,
            lr=grpo_config.r1_block_size_lr,
            exploration_rate=grpo_config.r1_exploration_rate,
            default_block_size=grpo_config.block_length,
        )
    if is_main_process():
        print(f"[R1] Block size candidates: {controller.candidates}")
        print(f"[R1] Exploration rate: {grpo_config.r1_exploration_rate}")
        print(f"[R1] Block size LR: {grpo_config.r1_block_size_lr}")
        if getattr(grpo_config, "r1_adaptive_proto", False):
            print(
                f"[R1] Adaptive DSCB prototypes: max={grpo_config.r1_max_prototypes}, "
                f"γ={grpo_config.r1_proto_gamma}, β_sim={grpo_config.r1_proto_beta_sim}, "
                f"min_stat_n={grpo_config.r1_proto_min_samples_stat}"
            )

    # Load model and tokenizer (same as run_train.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        import transformers.modeling_utils as _tmu

        _orig_caching_allocator_warmup = _tmu.caching_allocator_warmup

        def _safe_caching_allocator_warmup(model_to_load, *args, **kwargs):
            try:
                if getattr(model_to_load, "_tp_plan", None) is None:
                    model_to_load._tp_plan = []
            except Exception:
                pass
            return _orig_caching_allocator_warmup(model_to_load, *args, **kwargs)

        _tmu.caching_allocator_warmup = _safe_caching_allocator_warmup
    except Exception:
        pass

    # Some repos ship config.json without a registered `model_type`, so
    # AutoConfig.from_pretrained fails (e.g. diffusion-reasoning/LLaDA-8B-Instruct-SFT).
    # Pin specific model repos to known-good revisions to avoid accidental
    _PINNED_REVISIONS = {
        "inclusionAI/LLaDA2.1-mini": "bbb5715c881500b34234071e68dbf38c3d657c4e",
        "inclusionAI/LLaDA2.0-mini": "d23215abc5f5675daf171f6739d0386eab53f712",
    }
    _revision = _PINNED_REVISIONS.get(grpo_config.model_path)
    try:
        _cfg = AutoConfig.from_pretrained(
            grpo_config.model_path, trust_remote_code=True, revision=_revision
        )
    except ValueError as e:
        if is_main_process():
            print(
                f"[WARN] AutoConfig failed for {grpo_config.model_path}: {e}\n"
                f"       Falling back to PretrainedConfig to read config.json."
            )
        _cfg = PretrainedConfig.from_pretrained(
            grpo_config.model_path, trust_remote_code=True, revision=_revision
        )

    _auto_map = getattr(_cfg, "auto_map", {}) or {}
    _model_type = getattr(_cfg, "model_type", "")

    from rl.llada2_compat import _detect_rocm

    if _model_type == "llada2_moe":
        grpo_config.ddp_find_unused_parameters = True
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            revision=_revision,
        )
    elif _model_type == "sdar":
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            revision=_revision,
        )
    else:
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            revision=_revision,
        )

    if "AutoModelForCausalLM" in _auto_map:
        model = AutoModelForCausalLM.from_pretrained(**_load_kwargs).to(device)
    else:
        model = AutoModel.from_pretrained(**_load_kwargs).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True, revision=_revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    from rl.trainers.train_utils import get_mask_id as _get_mask_id

    grpo_config.mask_id = _get_mask_id(
        tokenizer=tokenizer, model=model, default=grpo_config.mask_id
    )
    if is_main_process():
        print(f"Using mask_id: {grpo_config.mask_id}")

    if not hasattr(model, "prepare_inputs_for_generation"):
        import types

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, model
        )

    # Apply model-specific patches (LLaDA2-MoE, SDAR, etc.)
    # Same patches as run_train.py for model compatibility
    if getattr(model.config, "model_type", "") == "llada2_moe":
        import functools

        _llada2_block_length = getattr(grpo_config, "block_length", 32)
        _inner = getattr(model, "model", model)
        _inner_cls = _inner.__class__
        _orig_inner_fwd = _inner_cls.forward

        @functools.wraps(_orig_inner_fwd)
        def _llada2_inner_forward(
            self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            if attention_mask is None:
                if input_ids is not None:
                    bs, sl = input_ids.shape[:2]
                    dev = input_ids.device
                elif inputs_embeds is not None:
                    bs, sl = inputs_embeds.shape[:2]
                    dev = inputs_embeds.device
                else:
                    return _orig_inner_fwd(
                        self, input_ids=input_ids, attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds, **kwargs,
                    )
                bl = _llada2_block_length
                n_blocks = (sl + bl - 1) // bl
                block_mask = torch.tril(torch.ones(n_blocks, n_blocks, device=dev))
                attention_mask = (
                    block_mask.repeat_interleave(bl, dim=0)[:sl, :]
                    .repeat_interleave(bl, dim=1)[:, :sl]
                    .unsqueeze(0).unsqueeze(0).log().to(torch.bfloat16).expand(bs, -1, -1, -1)
                )
            return _orig_inner_fwd(
                self, input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, **kwargs,
            )

        _inner_cls.forward = _llada2_inner_forward
        if is_main_process():
            print(f"Patched LLaDA2-MoE inner model (block_length={_llada2_block_length})")

    # Evaluation: create one R1-aware callback per domain.
    # Each callback uses the controller's current best block size for its domain
    # and logs to eval/accuracy/{domain_name} so all domains show on wandb.
    eval_callbacks = []
    eval_domains_added = set()

    for domain_name in domain_names:
        if domain_name in DATASET_MAP:
            val_dataset = DATASET_MAP[domain_name](
                tokenizer,
                subsample=SUB_SAMPLE_MAP.get(domain_name, 100),
                num_examples=0,
                add_reasoning=True,
            )
            eval_callbacks.append(
                _r1_per_domain_eval_callback(
                    domain_name,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    block_size_controller=controller,
                    gen_length=grpo_config.max_completion_length,
                    steps=grpo_config.diffusion_steps,
                    fallback_block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            )
            eval_domains_added.add(domain_name)

    # Guru-only training: monitor standard benchmarks during training (R1 routing uses
    # embedding → Q; these callbacks track held-out task accuracy).
    if "guru" in domain_names:
        for ed in GURU_TRAIN_EVAL_DOMAINS:
            if ed in eval_domains_added or ed not in DATASET_MAP:
                continue
            val_dataset = DATASET_MAP[ed](
                tokenizer,
                subsample=SUB_SAMPLE_MAP.get(ed, 100),
                num_examples=0,
                add_reasoning=True,
            )
            eval_callbacks.append(
                _r1_per_domain_eval_callback(
                    ed,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    block_size_controller=controller,
                    gen_length=grpo_config.max_completion_length,
                    steps=grpo_config.diffusion_steps,
                    fallback_block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            )
            eval_domains_added.add(ed)

    primary_eval_domain = domain_names[0]
    if primary_eval_domain not in DATASET_MAP:
        for cand in (*domain_names, *GURU_TRAIN_EVAL_DOMAINS):
            if cand in DATASET_MAP:
                primary_eval_domain = cand
                break

    primary_val_dataset = None
    if primary_eval_domain in DATASET_MAP:
        primary_val_dataset = DATASET_MAP[primary_eval_domain](
            tokenizer,
            subsample=SUB_SAMPLE_MAP.get(primary_eval_domain, 100),
            num_examples=0,
            add_reasoning=True,
        )

    # Configure LoRA
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )

    # Compatibility patch for _get_train_sampler
    try:
        import trl.trainer.grpo_trainer as _grpo_mod

        orig = getattr(_grpo_mod.GRPOTrainer, "_get_train_sampler", None)
        if orig is not None:
            def _wrapped_get_train_sampler(self, train_dataset=None):
                try:
                    return orig(self, train_dataset)
                except TypeError:
                    return orig(self)

            _grpo_mod.GRPOTrainer._get_train_sampler = _wrapped_get_train_sampler
    except Exception:
        pass

    # Create R1 trainer
    R1TrainerClass = get_r1_trainer_class(grpo_config.trainer_type)
    if is_main_process():
        print(f"[R1] Using trainer: {R1TrainerClass.__name__}")

    # Add R1 controller save callback (saves r1.json alongside every checkpoint)
    eval_callbacks.append(R1ControllerSaveCallback(controller))

    trainer = R1TrainerClass(
        args=grpo_config,
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        eval_dataset=primary_val_dataset,
        callbacks=eval_callbacks,
        block_size_controller=controller,
    )

    if is_main_process():
        # Avoid double init if HF Trainer / report_to also starts a W&B run on rank 0.
        if wandb.run is None:
            wandb.init(project=grpo_config.wandb_project, name=grpo_config.run_name)

    trainer.train()

    # Final save of r1.json (also saved at every save_steps via callback above)
    if is_main_process():
        import json
        final_path = os.path.join(grpo_config.output_dir, "r1.json")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "w") as f:
            json.dump(controller.state_dict(), f, indent=2)
        print(f"[R1] Final r1.json saved to {final_path}")

        print("\n[R1] Final block size preferences:")
        ctrl_kind = getattr(controller, "CONTROLLER_KIND", None)
        if ctrl_kind == AdaptiveProtoBlockController.CONTROLLER_KIND:
            print(
                f"  DSCB: active_prototypes={controller.active_prototype_count()} "
                f"(domain names like guru are not Q-table keys; slots are 0..K-1)"
            )
            for k in range(controller.max_prototypes):
                if not controller._slot_active(k):
                    continue
                pk = str(k)
                best_b = int(max(controller.q_values[pk], key=controller.q_values[pk].get))
                print(f"    proto {pk}: best_block_size={best_b}, Q={controller.q_values[pk]}")
        else:
            for domain in controller.domains:
                best_b = controller.get_best_block_size(domain)
                q_vals = controller.q_values[domain]
                has_centroid = controller.get_domain_centroid(domain) is not None
                print(
                    f"  {domain}: best_block_size={best_b}, "
                    f"centroid={'yes' if has_centroid else 'no'}, Q={q_vals}"
                )
        print(
            "\n[R1] r1.json is saved in each checkpoint-N/ dir and output_dir root.\n"
            "  For eval: --r1_controller_path checkpoints/.../checkpoint-500/r1.json"
        )


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    grpo_config.remove_unused_columns = False
    grpo_config.label_names = ["completion_ids"]
    main(grpo_config=grpo_config, model_config=model_config)
