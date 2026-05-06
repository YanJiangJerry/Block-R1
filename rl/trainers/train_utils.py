"""Utility functions for diffusion large language models training."""

import math
from typing import Any, Callable, Dict, List, MutableMapping, Optional

import torch


def grpo_group_normalized_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    std_eps: float = 1e-4,
    zero_std_threshold: float = 1e-6,
):
    """
    Standard GRPO-style advantages per prompt group: (r - mu_g) / (sigma_g + eps).

    When group std is below ``zero_std_threshold`` (e.g. identical completions),
    advantages are forced to zero instead of amplifying noise.

    This matches HF TRL's GRPO scaling and reduces multi-domain imbalance when
    batch-level softmax reweights samples (WD1 / D1 / GDPO / RevGRPO).
    """
    mean_g = rewards.view(-1, num_generations).mean(dim=1)
    std_g = rewards.view(-1, num_generations).std(dim=1)
    mean_rep = mean_g.repeat_interleave(num_generations, dim=0)
    std_rep = std_g.repeat_interleave(num_generations, dim=0)
    low = std_rep < zero_std_threshold
    advantages = (rewards - mean_rep) / (std_rep + std_eps)
    advantages = torch.where(low, torch.zeros_like(advantages), advantages)
    num_groups = std_g.numel()
    zero_std_count = (std_g < zero_std_threshold).sum().item()
    zero_std_ratio = zero_std_count / num_groups if num_groups > 0 else 0.0
    return advantages, std_rep, zero_std_ratio


def append_per_domain_reward_metrics(
    metrics_mode: MutableMapping[str, List],
    rewards: torch.Tensor,
    domain_per_sample_local: Optional[List[Any]],
    gather_object_fn: Callable,
) -> None:
    """
    Log mean final reward per domain for multi-domain (R1 / _route) training.

    ``rewards`` must already be ``gather``-ed across processes (same convention as
    ``train/reward``). ``domain_per_sample_local`` is the per-rank list of domain
    labels aligned with the local rows that were gathered into ``rewards``.
    WandB keys: ``rewards/domain/<domain_name>``.
    """
    if not domain_per_sample_local:
        return
    try:
        domains_all: List[Any] = gather_object_fn(domain_per_sample_local)
    except Exception:
        return
    if rewards.dim() != 1 or rewards.numel() != len(domains_all):
        return
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    r_cpu = rewards.detach().float().cpu()
    for i, d in enumerate(domains_all):
        key = str(d) if d is not None else "general"
        v = float(r_cpu[i].item())
        if math.isnan(v):
            continue
        sums[key] = sums.get(key, 0.0) + v
        counts[key] = counts.get(key, 0) + 1
    for key in sums:
        metrics_mode[f"rewards/domain/{key}"].append(sums[key] / counts[key])


# Known mask token ids per model family
_MASK_ID_BY_MODEL_TYPE = {
    "llada2_moe": 156895,  # inclusionAI/LLaDA2.x-mini
    "dream": 151666,  # Dream-org/Dream-v0-*
    "sdar": 151669,  # JetLM/SDAR-*, Gen-Verse/TraDo-* (Qwen2 vocab, <|MASK|> = 151669)
    # LLaDA (8B / 1.5) exposes mask_token_id via tokenizer, default fallback = 126336
}


def get_mask_id(tokenizer=None, model=None, default=126336):
    """
    Auto-detect mask_id from tokenizer or model config.

    Priority order:
      1. tokenizer.mask_token_id  (set by most HF tokenizers)
      2. model.config.mask_token_id  (set in some model configs)
      3. _MASK_ID_BY_MODEL_TYPE lookup on model.config.model_type
      4. ``default`` (126336 for LLaDA 8B / 1.5)

    Args:
        tokenizer: The tokenizer object (may have mask_token_id attribute)
        model: The model object (may have config.mask_token_id attribute)
        default: Default fallback value for LLaDA (126336)

    Returns:
        int: The mask token id

    Note:
        - LLaDA2-mini (llada2_moe) uses mask_token_id=156895
        - Dream model uses mask_token_id=151666
        - LLaDA 8B / 1.5 uses mask_token_id=126336
    """
    if tokenizer is not None:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            return tokenizer.mask_token_id

    if model is not None:
        # Unwrap PEFT wrapper if present
        cfg = None
        if hasattr(model, "config"):
            cfg = model.config
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            inner = model.base_model.model
            if hasattr(inner, "config"):
                cfg = inner.config

        if cfg is not None:
            if hasattr(cfg, "mask_token_id") and cfg.mask_token_id is not None:
                return cfg.mask_token_id
            # Fall back to per-model-type lookup (HF configs may use e.g. "Dream" vs "dream")
            model_type = str(getattr(cfg, "model_type", "") or "").lower()
            if model_type in _MASK_ID_BY_MODEL_TYPE:
                return _MASK_ID_BY_MODEL_TYPE[model_type]

    return default


def _dream_model_type_match(cfg) -> bool:
    if cfg is None:
        return False
    return str(getattr(cfg, "model_type", "") or "").lower() == "dream"


def is_dream_model(model):
    """
    Detect if the model is a Dream model (requires logits shift and SDPA mask shape).

    Handles DDP (``.module``), PEFT (``get_base_model`` / ``base_model.model``), and
    HuggingFace configs that set ``model_type`` to ``\"Dream\"`` or ``\"dream\"``.
    """
    m = model
    if hasattr(m, "module"):
        m = m.module

    def _check_one(inner) -> bool:
        if inner.__class__.__name__ == "DreamModel":
            return True
        return _dream_model_type_match(getattr(inner, "config", None))

    if _check_one(m):
        return True

    if hasattr(m, "get_base_model"):
        try:
            if _check_one(m.get_base_model()):
                return True
        except Exception:
            pass

    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        if _check_one(m.base_model.model):
            return True

    return False


def _iter_wrapped_model_candidates(model):
    """Yield candidate modules that may hold ``config`` (DDP / PEFT / base)."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    yield m
    if hasattr(m, "get_base_model"):
        try:
            yield m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        yield m.base_model.model


def model_uses_block_causal_attention_injection(model) -> bool:
    """
    True for families whose forward patch injects block-causal attention when
    ``attention_mask is None`` (same contract as ``wd1_grpo`` ``_forward_with_mask``
    and ``run_train.py`` SDAR / LLaDA2 patches).

    Passing a 2D HF padding mask disables that path and breaks diffusion generation.
    """
    for cand in _iter_wrapped_model_candidates(model):
        cfg = getattr(cand, "config", None)
        if cfg is None:
            continue
        mt = str(getattr(cfg, "model_type", "") or "").lower()
        if mt == "llada2_moe":
            return True
        if "sdar" in mt:
            return True
    return False


def _model_param_dtype(model) -> torch.dtype:
    """Dtype for SDPA bias / activations (match query). Unwraps DDP ``.module``.

    Prefer ``config.torch_dtype`` (matches bf16 training even when PEFT LoRA
    weights are fp32); else first floating-point parameter.
    """
    m = model
    if hasattr(m, "module"):
        m = m.module
    for cand in _iter_wrapped_model_candidates(model):
        cfg = getattr(cand, "config", None)
        if cfg is None:
            continue
        td = getattr(cfg, "torch_dtype", None)
        if isinstance(td, str):
            td = getattr(torch, td.rsplit(".", maxsplit=1)[-1], None)
        if td is not None and isinstance(td, torch.dtype) and td.is_floating_point:
            return td
    for p in m.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def prepare_diffusion_attention_mask(model, attention_mask):
    """
    Prepare ``attention_mask`` for diffusion RL forwards (aligned with
    ``wd1_grpo_trainer.RevDiffuGRPOTrainer._forward_with_mask`` and
    ``diffu_grpo_trainer.DiffuGRPOTrainer``).

    - **LLaDA2-MoE / SDAR (TraDo)**: return ``None`` so the model patch injects
      block-causal attention (a 2D padding mask would disable it).
    - **Dream**: convert 2D padding mask to SDPA key-padding bias ``[B, 1, 1, L]``.
    - **LLaDA 8B / 1.5 / others**: keep the standard 2D HF mask unchanged.
    """
    if attention_mask is None:
        return None
    if model_uses_block_causal_attention_injection(model):
        return None
    if not is_dream_model(model):
        return attention_mask
    if attention_mask.dtype == torch.bool:
        keep = attention_mask
    else:
        keep = attention_mask.bool()
    # SDPA requires attn_mask/bias dtype to match query (bf16 in training); float32
    # triggers: "invalid dtype for bias - should match query's dtype".
    dt = _model_param_dtype(model)
    min_val = torch.finfo(dt).min
    bias = (~keep).to(dtype=dt) * min_val
    return bias[:, None, None, :].to(device=attention_mask.device)


def apply_dream_logits_shift(logits, model):
    """
    Apply logits shift for Dream model if necessary.

    Dream model requires shifting logits: logits[:, :1] + logits[:, :-1]
    This is because Dream uses a different prediction paradigm than LLaDA.

    Args:
        logits: The logits tensor from model forward pass
        model: The model object (to detect if Dream model)

    Returns:
        torch.Tensor: Shifted logits if Dream model, unchanged otherwise
    """
    if is_dream_model(model):
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def _get_model_name_or_path(model):
    """Extract _name_or_path from model config, handling PEFT wrappers."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        name = getattr(cfg, "_name_or_path", "")
        if name:
            return name
    if hasattr(model, "base_model"):
        inner = getattr(model.base_model, "model", model.base_model)
        cfg = getattr(inner, "config", None)
        if cfg is not None:
            name = getattr(cfg, "_name_or_path", "")
            if name:
                return name
    if hasattr(model, "peft_config"):
        pc = model.peft_config
        if isinstance(pc, dict):
            for v in pc.values():
                name = getattr(v, "base_model_name_or_path", "")
                if name:
                    return name
    return ""


def is_llada2_mini(model):
    """Detect LLaDA2 mini models that tend to exhaust gen budget before producing answer tags."""
    name = _get_model_name_or_path(model).lower()
    return "llada2" in name and "mini" in name


def prefill_answer_structure(
    x, tokenizer, prompt_length, gen_length, mask_id, reserve_tokens=48
):
    """
    Pre-fill answer closing structure at the end of the generation sequence.

    For models that produce verbose reasoning and run out of budget before
    writing <answer> tags.  The pre-filled non-mask tokens survive diffusion
    denoising (``x0 = where(mask, x0, x)``), guaranteeing the structure.

    Layout::

        [...reasoning tokens...][</reasoning>\\n<answer>\\n][MASK * N][\\n</answer>]

    The MASK positions in the answer content zone are denoised normally so the
    model can still produce an actual answer.
    """
    answer_prefix = "\n</reasoning>\n<answer>\n"
    answer_suffix = "\n</answer>"

    prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(answer_suffix, add_special_tokens=False)

    prefix_len = len(prefix_ids)
    suffix_len = len(suffix_ids)

    answer_content_space = reserve_tokens - prefix_len - suffix_len
    if answer_content_space < 8:
        answer_content_space = 8
    total_reserve = prefix_len + answer_content_space + suffix_len

    seq_end = prompt_length + gen_length
    reserve_start = seq_end - total_reserve

    if reserve_start <= prompt_length:
        return x  # not enough space, skip

    prefix_tensor = torch.tensor(prefix_ids, dtype=x.dtype, device=x.device)
    x[:, reserve_start : reserve_start + prefix_len] = prefix_tensor

    suffix_tensor = torch.tensor(suffix_ids, dtype=x.dtype, device=x.device)
    x[:, seq_end - suffix_len : seq_end] = suffix_tensor

    return x
