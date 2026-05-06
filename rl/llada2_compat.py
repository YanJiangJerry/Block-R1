"""
Following dLLM-Factory's approach, we add compatibility utilities for LLaDA2-MoE models to enable seamless evaluation and generation without needing special-cased code paths in the main training/evaluation loops.
"""

import functools
import types
from typing import TypedDict
import numpy as np
import torch
import torch.nn.functional as F
import transformers.utils as _transformers_utils
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


def ensure_transformers_kwargs() -> None:
    if hasattr(_transformers_utils, "TransformersKwargs"):
        return

    class TransformersKwargs(TypedDict):
        pass

    _transformers_utils.TransformersKwargs = TransformersKwargs


def load_diffusion_model(
    model_path: str,
    *,
    revision: str | None = None,
    torch_dtype: torch.dtype,
    device: torch.device | None = None,
    quantization_config=None,
):
    ensure_transformers_kwargs()

    load_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if revision:
        load_kwargs["revision"] = revision
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, revision=revision
    )
    auto_map = getattr(config, "auto_map", {}) or {}
    model_cls = (
        AutoModelForCausalLM if "AutoModelForCausalLM" in auto_map else AutoModel
    )
    model = model_cls.from_pretrained(**load_kwargs)

    if device is not None:
        model = model.to(device)

    return model


def patch_prepare_inputs_for_generation(model):
    if hasattr(model, "prepare_inputs_for_generation"):
        return model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, model
    )
    return model


def patch_llada2_block_causal_attention(model, block_length: int = 32):
    """
    Patch LLaDA2-MoE models to add block-causal attention when no attention_mask is provided.
    This allows the same generation loop to work for LLaDA2-MoE during evaluation without needing special-cased code paths.
    """
    if getattr(model.config, "model_type", "") != "llada2_moe":
        return model

    inner_model = getattr(model, "model", model)
    if getattr(inner_model, "_llada2_block_causal_patched", False):
        return model

    original_forward = inner_model.forward

    @functools.wraps(original_forward)
    def wrapped_forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if attention_mask is None:
            if input_ids is not None:
                batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device
            elif inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
                device = inputs_embeds.device
            else:
                return original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )

            # R1 eval (or other callers) may set per-step block — must match eval/generate block_length.
            bl = int(getattr(self, "_llada2_eval_block_length", block_length))
            num_blocks = (seq_len + bl - 1) // bl
            block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
            attention_mask = (
                block_mask.repeat_interleave(bl, dim=0)[:seq_len, :]
                .repeat_interleave(bl, dim=1)[:, :seq_len]
                .unsqueeze(0)
                .unsqueeze(0)
                .log()
                .to(torch.bfloat16)
                .expand(batch_size, -1, -1, -1)
            )

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    inner_model.forward = types.MethodType(wrapped_forward, inner_model)
    inner_model._llada2_block_causal_patched = True
    inner_model._llada2_eval_block_length = int(block_length)
    return model


def set_llada2_eval_block_length(model, block_length: int) -> None:
    """Sync LLaDA2-MoE block-causal mask with the current generation block (e.g. R1-routed size)."""
    inner_model = getattr(model, "model", model)
    if getattr(inner_model, "_llada2_block_causal_patched", False):
        inner_model._llada2_eval_block_length = int(block_length)


def _detect_rocm() -> bool:
    """Detect ROCm platform using multiple heuristics."""
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return True
    try:
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0).lower()
            if any(kw in dev_name for kw in ("mi100", "mi200", "mi210", "mi250", "mi300", "instinct")):
                return True
    except Exception:
        pass
    return False


def patch_sdar_for_eval(model, block_length: int = 32):
    """Apply SDAR/TraDo-specific patches for correct eval generation.

    **All platforms**:
      1. Disable ``HAS_FLASH_ATTN`` flag.
      2. Replace ``SDARRMSNorm.forward`` with pure-PyTorch RMSNorm
         (triton version may produce garbage on ROCm).
      3. Wrap ``SDARModel.forward`` to inject a **block-causal** attention
         mask when ``attention_mask is None``.  SDAR models are trained
         with block-causal attention (bidirectional within each block,
         causal across blocks).  Using full bidirectional attention
         causes the model to see future MASK tokens and generate garbage.
      4. Disable ``fuse_cross_entropy``.

    **ROCm only**:
      5. Replace ``SDARAttention.forward`` with an SDPA version that manually
         expands KV heads and converts 2-D bool masks to 4-D additive masks.
    """
    import sys

    model_type = getattr(model.config, "model_type", "")
    if "sdar" not in model_type.lower():
        return model

    _is_rocm = _detect_rocm()

    # --- All platforms: replace flash_rms_norm and disable flash_attn ---------
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
            "disabled HAS_FLASH_ATTN for SDAR eval"
        )

        # --- ROCm only: replace attention with SDPA version -------------------
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

            print(
                "Applied ROCm attention patch for SDAR"
            )

    # --- All platforms: inject block-causal attention_mask when None -----------
    _inner = getattr(model, "model", model)
    _inner_cls = _inner.__class__
    _orig_inner_fwd = _inner_cls.forward

    _sdar_block_length = block_length

    @functools.wraps(_orig_inner_fwd)
    def _sdar_inner_forward(self, input_ids=None, attention_mask=None, **kwargs):
        if attention_mask is None and input_ids is not None:
            batch_size, seq_len = input_ids.shape[:2]
            device = input_ids.device
            bl = _sdar_block_length
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
    print(f"Patched SDAR inner model with block-causal attention (block_length={_sdar_block_length})")

    # --- All platforms: disable fuse_cross_entropy ----------------------------
    if getattr(model.config, "fuse_cross_entropy", False):
        model.config.fuse_cross_entropy = False
        print("Disabled fuse_cross_entropy for SDAR (eval needs logits)")

    return model


# ---------------------------------------------------------------------------
# Unified LLaDA2-MoE generation utilities (following dLLM-Factory approach)
# ---------------------------------------------------------------------------


def is_llada2_moe(model) -> bool:
    """Check if the (possibly DDP / PEFT-wrapped) model is LLaDA2-MoE."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    candidates = [m]
    if hasattr(m, "get_base_model"):
        try:
            candidates.append(m.get_base_model())
        except Exception:
            pass
    if hasattr(m, "base_model"):
        candidates.append(getattr(m.base_model, "model", m.base_model))
    for cand in candidates:
        cfg = getattr(cand, "config", None)
        if cfg is not None and getattr(cfg, "model_type", "") == "llada2_moe":
            return True
    return False


def build_block_causal_mask(batch_size: int, seq_len: int, block_length: int, device):
    """Create a block-causal 4D attention mask for LLaDA2-MoE.

    Tokens in block *i* attend to blocks 0..i; future blocks are masked with
    ``-inf``.  Shape: ``(batch_size, 1, seq_len, seq_len)`` in ``bfloat16``.
    """
    n_blocks = (seq_len + block_length - 1) // block_length
    bm = torch.tril(torch.ones(n_blocks, n_blocks, device=device))
    return (
        bm.repeat_interleave(block_length, dim=0)[:seq_len, :]
        .repeat_interleave(block_length, dim=1)[:, :seq_len]
        .unsqueeze(0)
        .unsqueeze(0)
        .log()
        .to(torch.bfloat16)
        .expand(batch_size, -1, -1, -1)
    )


def llada2_forward(model, input_ids, block_length=32):
    """Forward pass that adds block-causal attention for LLaDA2-MoE.

    For non-LLaDA2 models this is a plain ``model(input_ids)`` call.
    """
    if is_llada2_moe(model):
        mask = build_block_causal_mask(
            input_ids.shape[0], input_ids.shape[1], block_length, input_ids.device
        )
        return model(input_ids, attention_mask=mask)
    return model(input_ids)


# -- internal helpers used only inside generate_llada2 ----------------------


def _add_gumbel_noise(logits, temperature):
    """Gumbel-max categorical sampling noise (matches dLLM-Factory)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps):
    """Precompute number of tokens to unmask at each diffusion step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate_llada2(
    model,
    prompt,
    tokenizer=None,
    steps=128,
    gen_length=256,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=None,
):
    """LLaDA2-MoE generation — matches the proven ``rl/eval/generate.py`` loop.

    Uses the **same full-sequence diffusion logic** that works for every other
    model.  The block-causal attention mask is created automatically by the
    inner-model patch applied in ``run_train.py`` / ``eval.py`` (when
    ``attention_mask is None``).  We intentionally do **NOT** pass an explicit
    ``attention_mask`` so that the patch controls the mask creation — this is
    the same path that the standalone eval uses and avoids any interaction
    issues with quantised / PEFT-wrapped models.

    Args:
        model: LLaDA2-MoE model (plain, PEFT-wrapped, or quantised).
        prompt: ``(batch_size, prompt_length)`` input ids.
        tokenizer: Used for auto-detecting *mask_id* when it is ``None``.
        steps: Total diffusion denoising steps.
        gen_length: Number of tokens to generate.
        block_length: Semi-autoregressive block size.
        temperature: Gumbel noise temperature (0 → greedy).
        cfg_scale: Classifier-free guidance scale (0 → disabled).
        remasking: ``'low_confidence'`` or ``'random'``.
        mask_id: Mask token id (auto-detected if ``None``).

    Returns:
        ``(batch_size, prompt_length + gen_length)`` tensor with prompt +
        generated tokens.
    """
    from rl.trainers.train_utils import get_mask_id as _get_mask_id

    if mask_id is None:
        mask_id = _get_mask_id(tokenizer=tokenizer, model=model)

    batch_size, prompt_length = prompt.shape
    device = prompt.device

    # ---------- build initial sequence: [prompt | mask … mask] ----------
    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_length] = prompt.clone()

    prompt_index = x != mask_id

    # For LLaDA2 mini models: pre-fill answer structure to guarantee
    # the model produces <answer> tags before hitting the length limit.
    from rl.trainers.train_utils import is_llada2_mini, prefill_answer_structure

    if is_llada2_mini(model) and tokenizer is not None:
        x = prefill_answer_structure(x, tokenizer, prompt_length, gen_length, mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    # ---------- semi-autoregressive block loop ----------
    for num_block in range(num_blocks):
        start_idx = prompt_length + num_block * block_length
        end_idx = prompt_length + (num_block + 1) * block_length

        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )

        for i in range(steps_per_block):
            mask_index = x == mask_id

            # --- forward (NO explicit attention_mask → patch adds it) ---
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # --- sampling ---
            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # --- confidence for remasking ---
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=device)
            else:
                raise NotImplementedError(remasking)

            # Only allow unmasking within the current block
            x0_p[:, end_idx:] = -np.inf

            # Keep already-revealed tokens unchanged
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-np.inf, device=device)
            )

            # Transfer top-k tokens
            for j in range(confidence.shape[0]):
                num_tokens = num_transfer_tokens[j, i].item()
                if num_tokens > 0:
                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                    x[j, select_index] = x0[j, select_index]

    return x
