# import time
# from transformers import TrainerCallback

# class FlopsProfilerCallback(TrainerCallback):
#     def on_step_begin(self, args, state, control, model=None, **kwargs):
#         if state.global_step == 3:
#             try:
#                 from deepspeed.profiling.flops_profiler import FlopsProfiler

#                 self.prof = FlopsProfiler(model)
#                 self.prof.start_profile()
#                 self.start_time = time.time()
#                 print(
#                     f"\n[FLOPs Profiler] Started profiling at Global Step {state.global_step}..."
#                 )
#             except ImportError:
#                 print(
#                     "\n[FLOPs Profiler] DeepSpeed not found, skipping FLOPs calculation."
#                 )

#     def on_step_end(self, args, state, control, model=None, **kwargs):
#         if state.global_step in [3, 4] and hasattr(self, "prof"):
#             self.prof.stop_profile()
#             end_time = time.time()
#             duration = end_time - self.start_time

#             flops = self.prof.get_total_flops()
#             params = self.prof.get_total_params()

#             print(f"\n{'='*40}")
#             print(f"[FLOPs Profiler] Step {state.global_step} Stats:")
#             print(f"  Total FLOPs: {flops:.2e}")
#             print(f"  Time Cost  : {duration:.2f} s")
#             if duration > 0:
#                 print(f"  Throughput : {flops / duration:.2e} FLOPs/s")
#             print(f"{'='*40}\n")

#             self.prof.end_profile()
#             del self.prof


import os
import warnings
import logging
import inspect


# Set PYTHONWARNINGS env var so subprocesses and early imports respect it
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Ignore all warnings via the warnings module
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


# Replace warnings.showwarning and warnings.warn with no-ops to silence any direct calls
def _noop_showwarning(*_args, **_kwargs):
    return


def _noop_warn(*_args, **_kwargs):
    return


warnings.showwarning = _noop_showwarning
warnings.warn = _noop_warn
# Configure logging to show only ERROR or higher to silence warning-level logs
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
# Additionally silence noisy libraries explicitly (optional)
for _name in [
    "transformers",
    "torch",
    "pydantic",
    "torch.distributed.run",
    "accelerate",
]:
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass

import torch
import torch.distributed as dist
import wandb

from data_utils import (
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
    # Dynamically inspect the __init__ method to find valid parameters
    valid_params = set(inspect.signature(_orig_lora_init).parameters.keys())

    # Filter the kwargs: keep only keys that exist in valid_params
    clean_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    # Call the original __init__ with the sanitized arguments
    _orig_lora_init(self, *args, **clean_kwargs)


# Apply the patch
LoraConfig.__init__ = _robust_lora_init

# ---------------------------------------------------------------------------
# Compatibility shim: inject TransformersKwargs into transformers.utils if the
# installed version does not provide it (required by LLaDA2.x-mini remote code).
# This is a no-op when TransformersKwargs already exists (newer transformers).
# ---------------------------------------------------------------------------
import transformers.utils as _transformers_utils

if not hasattr(_transformers_utils, "TransformersKwargs"):
    from typing import TypedDict

    class TransformersKwargs(TypedDict):
        pass

    _transformers_utils.TransformersKwargs = TransformersKwargs

from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correctness_reward_func_math,
    countdown_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    sudoku_reward_func,
    xmlcount_reward_func,
    block_format_reward,
    code_reward_func,
    get_code_format_reward,
    code_reward,
    mc_reward_func,
    knights_knaves_reward_func,
)

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ModelConfig, TrlParser

from rl.trainers.diffu_grpo_config import DiffuGRPOConfig

from rl.trainers.diffu_grpo_trainer import DiffuGRPOTrainer
from rl.trainers.stable_drl_trainer import StableDRLTrainer
from rl.trainers.eval_callback import AccuracyEvalCallback
from rl.trainers.rev_grpo_ref_pol_trainer import RevDiffuRefPolGRPOTrainer
from rl.trainers.wd1_grpo_trainer import RevDiffuGRPOTrainer
from rl.trainers.rev_grpo_trainer_psr import RevPSRDiffuGRPOTrainer
from rl.trainers.gdpo_trainer import GDPOTrainer
from rl.trainers.mdpo_trainer import MDPOTrainer
from rl.trainers.espo_trainer import ESPOTrainer

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


# Evaluation data from eval/*.py
DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "mbpp": MBPPDataset,
    "humaneval": HumanEvalDataset,
    "kodcode": KodCodeDataset,  # Default use last 500 samples of KodCode for testing
    "mmlu": MMLUDataset,
    "mmlu_pro": MMLUProDataset,
    "hellaswag": HellaSwagDataset,
    "arc_c": ARCCDataset,
    "arc_e": ARCEDataset,
    "gpqa": GPQADataset,
    "knights_and_knaves": KnightsAndKnavesDataset,
}
# Quick validation random subsample sizes
# SUB_SAMPLE_MAP = {
#     "gsm8k": 500,
#     "math": -1,
#     "countdown": -1,
#     "sudoku": -1,
#     "mbpp": -1,
#     "humaneval": -1,
#     "kodcode": 500,
#     "mmlu": 500,
#     "mmlu_pro": 500,
#     "hellaswag": 500,
#     "arc_c": 500,
#     "arc_e": 500,
# }
# Default to 500 samples for quick validation during development; set to -1 for full evaluation
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


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main(grpo_config, model_config):
    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    set_trainer_type(grpo_config.trainer_type)

    # Load dataset and reward functions
    val_dataset = None
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        # Format reward + correctness reward
        reward_functions = [
            xmlcount_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]

    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        # Format reward + correctness reward
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        # Small data for quick test
        reward_functions = [
            countdown_reward_func,
        ]

    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        # Small data for quick test
        reward_functions = [
            sudoku_reward_func,
        ]

    elif grpo_config.dataset == "mbpp":
        # Use MBPP's own training set (task_id 511-974)
        # dataset = get_mbpp_questions()
        # code_format = get_code_format_reward(language="python")
        # reward_functions = [code_format, code_reward]

        # Use KodCode as training set (larger), MBPP for evaluation
        dataset = get_kodcode_light_rl_10k("train")
        code_format = get_code_format_reward(language="python")
        reward_functions = [code_format, code_reward]

    elif grpo_config.dataset == "humaneval":
        # Use KodCode as training set (larger), HumanEval for evaluation
        dataset = get_kodcode_light_rl_10k("train")
        code_format = get_code_format_reward(language="python")
        reward_functions = [code_format, code_reward]

    elif grpo_config.dataset == "kodcode":
        dataset = get_kodcode_light_rl_10k("train")
        code_format = get_code_format_reward(language="python")
        reward_functions = [code_format, code_reward]

    elif grpo_config.dataset == "mmlu":
        dataset = get_mmlu_questions(
            "auxiliary_train"
        )  # Use auxiliary_train for training
        reward_functions = [mc_reward_func]

    elif grpo_config.dataset == "mmlu_pro":
        # MMLU-Pro has only 70 validation samples, use MMLU's auxiliary_train for training
        # Evaluation will still be on MMLU-Pro test set
        dataset = get_mmlu_questions("auxiliary_train")
        reward_functions = [mc_reward_func]

    elif grpo_config.dataset == "hellaswag":
        dataset = get_hellaswag_questions("train")
        reward_functions = [mc_reward_func]

    elif grpo_config.dataset == "arc_c":
        dataset = get_arc_c_questions("train")
        reward_functions = [mc_reward_func]

    elif grpo_config.dataset == "arc_e":
        dataset = get_arc_e_questions("train")
        reward_functions = [mc_reward_func]

    elif grpo_config.dataset == "gpqa":
        # GPQA has only 448 examples (train split only), use MMLU auxiliary_train for RL training
        # Evaluation will be on the GPQA test set
        dataset = get_mmlu_questions("auxiliary_train")
        reward_functions = [mc_reward_func]

    elif grpo_config.dataset == "knights_and_knaves":
        dataset = get_knights_and_knaves_questions("train")
        reward_functions = [knights_knaves_reward_func]

    # For b1_wll and b1_d1, add block format reward
    # if grpo_config.trainer_type in ["b1_d1", "b1_wll"]:
    #     reward_functions.append(block_format_reward)

    # Shuffle dataset with fixed seed for reproducibility (except mbpp which has fixed train split)
    # if grpo_config.dataset != "mbpp":
    #     dataset = dataset.shuffle(seed=grpo_config.seed)
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split training set
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))
    elif grpo_config.dataset == "mbpp":
        # MBPP default training set: task_id 511-974 (indices 500-973 in 0-indexed)
        train_set = dataset.select(range(500, 974))
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
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
                    # prefer setting an instance attribute to avoid changing
                    # class semantics for other models
                    model_to_load._tp_plan = []
            except Exception:
                # If we can't set it, fall through and let the original
                # function handle or raise a clearer error
                pass
            return _orig_caching_allocator_warmup(model_to_load, *args, **kwargs)

        _tmu.caching_allocator_warmup = _safe_caching_allocator_warmup
    except Exception:
        # If monkeypatching fails for any reason, continue without it and
        # allow the original error to surface (so the user can see the
        # underlying issue). This keeps behavior deterministic.
        pass

    # Load model and tokenizer - use SFT path if on top of SFT
    # Use AutoModelForCausalLM for models that register it (needed for LLaDA2-MoE
    # where AutoModel returns the base model without LM head). Fall back to
    # AutoModel only for models that don't register AutoModelForCausalLM (e.g. Dream).
    
    # Some HuggingFace repos may ship a minimal config.json without `model_type`,
    # which makes `AutoConfig.from_pretrained()` fail (e.g. diffusion-reasoning/LLaDA-8B-Instruct-SFT).
    # In that case we fall back to the generic PretrainedConfig loader to keep
    # the rest of the pipeline unchanged for all other models.
    from transformers import AutoConfig, PretrainedConfig

    # Pin specific model repos to known-good revisions to avoid accidental
    # Update at 2026.4 to maintain the stable commit of LLaDA2-MoE and LLaDA2.0-mini.
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

    # LLaDA2-MoE models suffer severe quality degradation under 4-bit NF4
    # quantisation (the MoE gating layer loses too much precision, causing
    # many generation positions to collapse to token-0 = "!").  Disable
    # quantisation for these models and load in bf16 instead.
    _model_type = getattr(_cfg, "model_type", "")
    # MoE routing can lead to parameters being unused on some ranks for a given step.
    # Enable unused-parameter detection only for LLaDA2-MoE to keep other models unchanged.
    if _model_type == "llada2_moe":
        grpo_config.ddp_find_unused_parameters = True
    from rl.llada2_compat import _detect_rocm
    _is_rocm_early = _detect_rocm()
    if _model_type == "llada2_moe":
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            revision=_revision,
        )
        print(
            f"[INFO] LLaDA2-MoE detected — loading in bf16 (4-bit quant disabled "
            f"to avoid MoE gating degradation)"
        )
    elif _model_type == "sdar":
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            revision=_revision,
        )
        print(
            f"[INFO] SDAR detected — loading in bf16 (4-bit quant "
            f"disabled to avoid NF4 degradation)"
        )
    else:
        _load_kwargs = dict(
            pretrained_model_name_or_path=grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            revision=_revision,
        )

    # Model loading logic:
    if "AutoModelForCausalLM" in _auto_map:
        model = AutoModelForCausalLM.from_pretrained(**_load_kwargs).to(device)
    else:
        model = AutoModel.from_pretrained(**_load_kwargs).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True, revision=_revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Auto-detect and set mask_id.
    # Priority: tokenizer.mask_token_id -> model.config.mask_token_id
    #           -> model_type lookup (LLaDA2-mini: 156895, Dream: 151666) -> default 126336
    from rl.trainers.train_utils import get_mask_id as _get_mask_id

    grpo_config.mask_id = _get_mask_id(
        tokenizer=tokenizer, model=model, default=grpo_config.mask_id
    )
    print(
        f"Using mask_id: {grpo_config.mask_id} (model_type={getattr(model.config, 'model_type', 'unknown')})"
    )

    # Patch for Dream model: add prepare_inputs_for_generation if missing (required by PEFT)
    if not hasattr(model, "prepare_inputs_for_generation"):

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        import types

        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, model
        )
        print("Patched model with prepare_inputs_for_generation for PEFT compatibility")

    # ---------------------------------------------------------------------------
    # Patch for SDAR models:
    # All platforms:
    #   1. Disable HAS_FLASH_ATTN so the attention layers fall through to SDPA.
    #   2. Replace flash_rms_norm with standard PyTorch RMSNorm (the triton
    #      kernel from flash_attn produces garbage on ROCm; the standard impl
    #      is safe and correct everywhere).
    #   3. Inject an all-ones attention_mask when it is missing.
    # ROCm additionally:
    #   4. Patch SDARAttention.forward to expand a 2-D bool mask into a proper
    #      4-D mask before calling F.scaled_dot_product_attention (SDPA cannot
    #      broadcast a raw (B, S) mask to (B, H, S, S) on its own).
    # ---------------------------------------------------------------------------
    if "sdar" in getattr(model.config, "model_type", "").lower():
        import functools
        import sys

        from rl.llada2_compat import _detect_rocm
        _is_rocm = _detect_rocm()

        _inner_tmp = getattr(model, "model", model)
        _sdar_mod = sys.modules.get(_inner_tmp.__class__.__module__)
        if _sdar_mod:
            if hasattr(_sdar_mod, "HAS_FLASH_ATTN"):
                _sdar_mod.HAS_FLASH_ATTN = False

            _rms_cls = getattr(_sdar_mod, "SDARRMSNorm", None)
            if _rms_cls is not None:

                def _torch_rms_norm_forward(self, hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(
                        variance + self.variance_epsilon
                    )
                    return self.weight * hidden_states.to(input_dtype)

                _rms_cls.forward = _torch_rms_norm_forward

            print(
                "Replaced SDARRMSNorm with PyTorch RMSNorm, "
                "disabled HAS_FLASH_ATTN for SDAR"
            )

            if _is_rocm:
                _attn_cls = getattr(_sdar_mod, "SDARAttention", None)
                _apply_rotary = getattr(_sdar_mod, "apply_rotary_pos_emb", None)
                if _attn_cls is not None and _apply_rotary is not None:

                    def _rocm_attn_forward(
                        self,
                        hidden_states,
                        position_embeddings,
                        attention_mask=None,
                        past_key_value=None,
                        cache_position=None,
                        **kwargs,
                    ):
                        input_shape = hidden_states.shape[:-1]
                        bsz, q_len = input_shape
                        hidden_shape = (*input_shape, -1, self.head_dim)

                        query_states = self.q_norm(
                            self.q_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                        key_states = self.k_norm(
                            self.k_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                        value_states = (
                            self.v_proj(hidden_states)
                            .view(hidden_shape)
                            .transpose(1, 2)
                        )

                        cos, sin = position_embeddings
                        query_states, key_states = _apply_rotary(
                            query_states, key_states, cos, sin
                        )

                        if past_key_value is not None and kwargs.get("store_kv", False):
                            key_states, value_states = past_key_value.update(
                                key_states, value_states, self.layer_idx
                            )
                        elif (
                            past_key_value is not None
                            and not kwargs.get("store_kv", False)
                            and len(past_key_value) > self.layer_idx
                        ):
                            pk, pv = past_key_value[self.layer_idx]
                            key_states = torch.cat([pk, key_states], dim=-2)
                            value_states = torch.cat([pv, value_states], dim=-2)

                        n_rep = self.num_key_value_groups
                        if n_rep > 1:
                            key_states = (
                                key_states[:, :, None, :, :]
                                .expand(-1, -1, n_rep, -1, -1)
                                .reshape(
                                    bsz,
                                    self.num_attention_heads,
                                    key_states.shape[-2],
                                    self.head_dim,
                                )
                            )
                            value_states = (
                                value_states[:, :, None, :, :]
                                .expand(-1, -1, n_rep, -1, -1)
                                .reshape(
                                    bsz,
                                    self.num_attention_heads,
                                    value_states.shape[-2],
                                    self.head_dim,
                                )
                            )

                        attn_mask = None
                        if attention_mask is not None:
                            k_len = key_states.shape[-2]
                            am = attention_mask
                            if am.dim() == 2:
                                am = am[:, None, None, :k_len]
                            elif am.dim() == 4:
                                am = am[:, :, :, :k_len]
                            if am.dtype == torch.bool:
                                fmin = torch.finfo(query_states.dtype).min
                                z = torch.zeros(
                                    (),
                                    dtype=query_states.dtype,
                                    device=am.device,
                                )
                                ni = torch.full(
                                    (),
                                    fmin,
                                    dtype=query_states.dtype,
                                    device=am.device,
                                )
                                attn_mask = torch.where(am, z, ni)
                            else:
                                attn_mask = am.to(query_states.dtype)

                        attn_output = torch.nn.functional.scaled_dot_product_attention(
                            query=query_states,
                            key=key_states,
                            value=value_states,
                            attn_mask=attn_mask,
                            is_causal=False,
                            scale=self.scaling,
                        )
                        attn_output = attn_output.transpose(1, 2).contiguous()
                        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                        attn_output = self.o_proj(attn_output)
                        return attn_output, None

                    _attn_cls.forward = _rocm_attn_forward

                print("Applied ROCm attention patch for SDAR")

        _inner = getattr(model, "model", model)
        _inner_cls = _inner.__class__
        _orig_inner_fwd = _inner_cls.forward
        _sdar_bl = getattr(grpo_config, "block_length", 32)

        @functools.wraps(_orig_inner_fwd)
        def _sdar_inner_forward(self, input_ids=None, attention_mask=None, **kwargs):
            if attention_mask is None and input_ids is not None:
                batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device
                bl = _sdar_bl
                num_blocks = (seq_len + bl - 1) // bl
                bm = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
                attention_mask = (
                    bm.repeat_interleave(bl, dim=0)[:seq_len, :]
                    .repeat_interleave(bl, dim=1)[:, :seq_len]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1, -1)
                )
            return _orig_inner_fwd(
                self, input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

        _inner_cls.forward = _sdar_inner_forward
        print(f"Patched SDAR inner model with block-causal attention (block_length={_sdar_bl})")

        # Disable fuse_cross_entropy: when True the model returns logits=None
        # during training mode, but the RL trainer needs logits for generation.
        if getattr(model.config, "fuse_cross_entropy", False):
            model.config.fuse_cross_entropy = False
            print(
                "Disabled fuse_cross_entropy for SDAR (RL trainer needs logits during generation)"
            )

    # ---------------------------------------------------------------------------
    # Patch for LLaDA2-MoE models: the inner LLaDA2MoeModel.forward crashes
    # when attention_mask is None (which happens during diffusion generation).
    # LLaDA2-MoE uses block-causal attention: tokens in block i attend to
    # blocks 0..i but NOT future blocks. The official generate code creates
    # this mask explicitly; we replicate it here when attention_mask is None.
    # ---------------------------------------------------------------------------
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
                        self,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        **kwargs,
                    )
                # Build block-causal mask matching the official LLaDA2 generate:
                #   block_mask = tril(ones(n_blocks, n_blocks))
                #   expanded  = repeat_interleave → (1,1,sl,sl) → .log() → bfloat16
                # Result: 0 where attend, -inf where masked.
                bl = _llada2_block_length
                n_blocks = (sl + bl - 1) // bl
                block_mask = torch.tril(torch.ones(n_blocks, n_blocks, device=dev))
                attention_mask = (
                    block_mask.repeat_interleave(bl, dim=0)[:sl, :]
                    .repeat_interleave(bl, dim=1)[:, :sl]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .log()
                    .to(torch.bfloat16)
                    .expand(bs, -1, -1, -1)
                )
            return _orig_inner_fwd(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        _inner_cls.forward = _llada2_inner_forward
        print(
            f"Patched LLaDA2-MoE inner model with block-causal attention "
            f"(block_length={_llada2_block_length})"
        )

    # if grpo_config.trainer_type == "b1_wll":
    #     # Ensure the special token "\\block" exists in the tokenizer. If not, add it and resize the model embeddings.
    #     token_str = "\\block"
    #     block_id = None

    #     try:
    #         # add_special_tokens returns the number of tokens added (>=0)
    #         tokenizer.add_special_tokens({"additional_special_tokens": [token_str]})
    #         # resize model embeddings to include the new token
    #         try:
    #             model.resize_token_embeddings(len(tokenizer))
    #         except Exception:
    #             # Some model wrappers may not expose resize_token_embeddings; ignore
    #             pass
    #         # read back id
    #         try:
    #             res = tokenizer.convert_tokens_to_ids([token_str])
    #             block_id = res[0] if isinstance(res, (list, tuple)) else res
    #         except Exception:
    #             block_id = getattr(grpo_config, "mask_id", None)
    #         print(f"Added special token {token_str} with id {block_id}")
    #         tokenizer.block_token = token_str
    #         # 126349
    #         tokenizer.block_token_id = block_id
    #     except Exception as e:
    #         # Fallback: use mask_id from config if available
    #         block_id = getattr(grpo_config, "mask_id", None)
    #         print(
    #             f"Could not add special token {token_str} due to {e}; falling back to mask_id={block_id}"
    #         )

    val_dataset = DATASET_MAP[grpo_config.dataset](
        tokenizer,
        subsample=SUB_SAMPLE_MAP[grpo_config.dataset],
        num_examples=0,
        add_reasoning=True,  # prefill for all models
    )

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Compatibility patch: some versions of transformers call
    # Trainer._get_train_sampler(train_dataset) while older TRL's
    # GRPOTrainer defines _get_train_sampler(self) without the dataset
    # argument. If that's the case, wrap the original method so it
    # accepts the extra parameter and forwards to the implementation.
    try:
        import trl.trainer.grpo_trainer as _grpo_mod

        orig = getattr(_grpo_mod.GRPOTrainer, "_get_train_sampler", None)

        if orig is not None:

            def _wrapped_get_train_sampler(self, train_dataset=None):
                # The original implementation may only expect (self,), so
                # call it accordingly. If it already accepts train_dataset,
                # Python will bind it fine when orig is the function object
                try:
                    return orig(self, train_dataset)
                except TypeError:
                    # Fallback to calling without the extra argument
                    return orig(self)

            _grpo_mod.GRPOTrainer._get_train_sampler = _wrapped_get_train_sampler
    except Exception:
        # If monkeypatching fails, let the error surface later so the
        # user can see the underlying incompatibility.
        pass
    if is_main_process():
        print("Trainer type is: ", grpo_config.trainer_type)

    # Initialize and run trainer
    if grpo_config.trainer_type == "wd1" or grpo_config.trainer_type == "b1_wll":
        # NSR + PSR + d1 objective = wd1's RevGRPO trainer with negative reference samples
        trainer = RevDiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "wd1_pos":
        trainer = RevPSRDiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "d1" or grpo_config.trainer_type == "b1_d1":
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "stable_drl" or grpo_config.trainer_type == "b1_stable_drl":
        trainer = StableDRLTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "wd1_ref":
        # add reference policy regularisation for
        trainer = RevDiffuRefPolGRPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "gdpo" or grpo_config.trainer_type == "b1_gdpo":
        trainer = GDPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "mdpo" or grpo_config.trainer_type == "b1_mdpo":
        trainer = MDPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "espo" or grpo_config.trainer_type == "b1_espo":
        trainer = ESPOTrainer(
            args=grpo_config,
            model=model,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    else:
        raise Exception("Not know trainer type")

    if is_main_process():
        wandb.init(project=grpo_config.wandb_project, name=grpo_config.run_name)

    # # Add FLOPs profiler callback
    # trainer.add_callback(FlopsProfilerCallback())

    # To _inner_training_loop() in GRPOTrainer
    # 1. prepare input 2. forward 3. compute rewards 4. compute loss 5. backward 6. optimizer step 7. evaluate
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    grpo_config.remove_unused_columns = False
    grpo_config.label_names = ["completion_ids"]
    main(grpo_config=grpo_config, model_config=model_config)
