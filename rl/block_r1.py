"""
Block-R1 utilities.
1) Stage-2 multi-block evaluation on *training* splits with per-domain rewards.
2) build_block_r1: filter examples where reward_A > reward_B at the block that maximizes (A-B),
   stream a multi-domain train.jsonl (+ META.json) for downstream training (CPU-only, bounded RAM).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import datetime


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-") or "model"


def _init_dist() -> Tuple[int, int, int]:
    """
    Best-effort distributed init compatible with torchrun/accelerate.
    Returns (rank, world_size, local_rank).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    # Bind this process to its GPU before NCCL init to avoid "device unknown" / missing
    # device_id warnings (see PyTorch distributed docs).
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            # PyTorch process group timeout (seconds). Default is 600s on many builds and can be
            # too small for long-running/uneven inference batches, leading to watchdog timeouts.
            _timeout_env = os.environ.get("TORCH_DIST_TIMEOUT_SEC", "").strip()
            timeout = (
                datetime.timedelta(seconds=int(_timeout_env))
                if _timeout_env
                else None
            )
            # Match rl/eval/eval.py: device_id silences "No device id is provided" / NCCL mapping warnings.
            if backend == "nccl":
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method="env://",
                    device_id=torch.device(f"cuda:{local_rank}"),
                    **({"timeout": timeout} if timeout is not None else {}),
                )
            else:
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method="env://",
                    **({"timeout": timeout} if timeout is not None else {}),
                )
    rank = (
        torch.distributed.get_rank()
        if (torch.distributed.is_available() and torch.distributed.is_initialized())
        else 0
    )
    world_size = (
        torch.distributed.get_world_size()
        if (torch.distributed.is_available() and torch.distributed.is_initialized())
        else 1
    )
    return rank, world_size, local_rank



def _safe_destroy_process_group() -> None:
    """Best-effort teardown so NCCL does not warn about leaked process groups."""
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    try:
        torch.distributed.destroy_process_group()
    except Exception:
        pass


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _jsonl_write(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl_append(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl_count_rows(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _resume_index_in_local_positions(part_path: str, local_positions: List[int]) -> int:
    """
    How many leading entries of local_positions already have at least one row in part_path.

    Uses unique ``pos`` values (not raw line count) so duplicate lines from interrupted
    runs do not skip ahead and leave gaps.
    """
    seen_pos: set[int] = set()
    for row in _jsonl_iter(part_path):
        seen_pos.add(int(row["pos"]))
    for i, p in enumerate(local_positions):
        if int(p) not in seen_pos:
            return i
    return len(local_positions)


def _jsonl_iter(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _jsonl_read(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_reward_norm_map(
    out_root: str, model_safe: str, domain: str, block_sizes: List[int]
) -> Dict[Tuple[str, int], float]:
    """Line-by-line load; last row wins on duplicate (example_id, block)."""
    m: Dict[Tuple[str, int], float] = {}
    for b in block_sizes:
        p = os.path.join(out_root, model_safe, f"rewards_{domain}_b{b}.jsonl")
        for row in _jsonl_iter(p):
            m[(str(row["example_id"]), int(b))] = float(row["reward_norm"])
    return m


def _json_safe_value(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe_value(v[k]) for k in v}
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    return str(v)


def _train_row_to_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe_value(ex[k]) for k in ex}


def _dedup_pick_text(row: Dict[str, Any]) -> Optional[str]:
    """
    Pick a "question/prompt" string for substring-based de-dup.
    Prefer common keys but fall back gracefully.
    """
    for k in ("prompt", "question", "problem", "input"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _dedup_substring_inplace_jsonl(train_path: str) -> Dict[str, int]:
    """
    Remove duplicated samples across sources:
    - if one question/prompt is a substring of another, remove the shorter one.
    - deterministic: prefer keeping the longer text; if equal length, keep the first-seen.

    Implementation notes:
    - Avoid O(N^2) comparisons by using a cheap n-gram candidate index (k=48) based on
      first/middle/last windows.
    - Rewrites train.jsonl in-place via tmp file.
    """
    rows: List[Dict[str, Any]] = _jsonl_read(train_path)
    texts: List[Optional[str]] = [_dedup_pick_text(r) for r in rows]

    # Only rows with a comparable text participate in substring dedup.
    idxs = [i for i, t in enumerate(texts) if isinstance(t, str) and t]
    if len(idxs) <= 1:
        return {"removed": 0, "kept": len(rows), "scanned": len(rows)}

    # Sort by text length descending so we keep longer first.
    idxs.sort(key=lambda i: (len(texts[i] or ""), -i), reverse=True)

    k = 48
    gram_to_kept: Dict[str, List[int]] = {}
    kept: List[int] = []
    keep_mask = [True] * len(rows)

    def grams(s: str) -> List[str]:
        s = s.strip()
        if len(s) <= k:
            return [s]
        mid = len(s) // 2
        return [s[:k], s[max(0, mid - k // 2) : max(0, mid - k // 2) + k], s[-k:]]

    for i in idxs:
        s = texts[i]
        if not isinstance(s, str) or not s:
            continue
        s_norm = s.strip()

        # Find candidate kept strings that might contain s_norm.
        cand: set = set()
        for g in grams(s_norm):
            for j in gram_to_kept.get(g, []):
                cand.add(j)

        # If any kept longer string contains this string, drop this one (shorter or equal).
        dropped = False
        for j in cand:
            tj = texts[j]
            if not isinstance(tj, str) or not tj:
                continue
            if len(tj) < len(s_norm):
                continue
            if s_norm in tj:
                keep_mask[i] = False
                dropped = True
                break

        if dropped:
            continue

        kept.append(i)
        for g in grams(s_norm):
            gram_to_kept.setdefault(g, []).append(i)

    removed = sum(1 for x in keep_mask if not x)
    if removed == 0:
        return {"removed": 0, "kept": len(rows), "scanned": len(rows)}

    tmp_path = train_path + ".dedup.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if keep_mask[i]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, train_path)
    return {"removed": removed, "kept": len(rows) - removed, "scanned": len(rows)}


def _maybe_set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_train_dataset(domain: str):
    # Use the same loaders as training.
    from rl.data_utils import (
        get_countdown_questions,
        get_gsm8k_questions,
        get_kodcode_light_rl_10k,
        get_knights_and_knaves_questions,
        get_math_questions,
        get_sudoku_questions,
        set_trainer_type,
    )

    # R1 doesn't require \\block markers; use baseline prompts.
    set_trainer_type("wd1")

    if domain == "gsm8k":
        return get_gsm8k_questions("train")
    if domain == "math":
        return get_math_questions("train")
    if domain == "countdown":
        return get_countdown_questions("train")
    if domain == "sudoku":
        return get_sudoku_questions()
    if domain == "kodcode":
        return get_kodcode_light_rl_10k("train")
    if domain == "knights_and_knaves":
        return get_knights_and_knaves_questions("train")
    raise ValueError(f"Unknown domain: {domain}")


def _subsample_fixed(ds, n: int, seed: int):
    """
    Deterministic across runs and ranks:
    - shuffle with fixed seed
    - take first min(n, len(ds)) (domains differ; e.g. gsm8k train is 7473)
    """
    n_eff = min(int(n), len(ds))
    # Record the original row index so we can persist the sampled subset and
    # keep stable IDs across reruns (given the same cached dataset snapshot).
    try:
        if "_orig_idx" not in getattr(ds, "column_names", []):
            ds = ds.add_column("_orig_idx", list(range(len(ds))))
    except Exception:
        # If adding a column fails for any reason, fall back to position-based IDs.
        pass
    ds = ds.shuffle(seed=seed)
    return ds.select(range(n_eff))


def _extract_prompt_text(example: Dict[str, Any], tokenizer) -> str:
    from trl.data_utils import maybe_apply_chat_template

    out = maybe_apply_chat_template(example, tokenizer)
    # TRL returns {"prompt": "..."} for chat templates
    if isinstance(out, dict) and "prompt" in out:
        return out["prompt"]
    # Fallback: raw prompt list
    p = example.get("prompt")
    if isinstance(p, list) and p and isinstance(p[0], dict):
        # naive concatenation if template missing
        return "\n".join(str(m.get("content", "")) for m in p)
    return str(p)


def _tokenize_prompts(
    tokenizer,
    prompts_text: List[str],
    device: torch.device,
    *,
    max_prompt_tokens: Optional[int] = None,
):
    """
    Tokenize with left padding. Optionally truncate very long prompts to avoid
    pathological slowdowns (some domains like knights_and_knaves can be much longer).
    """
    tok = tokenizer(
        prompts_text,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        truncation=(max_prompt_tokens is not None),
        max_length=(int(max_prompt_tokens) if max_prompt_tokens is not None else None),
    )
    return {k: v.to(device) for k, v in tok.items()}


def _apply_llada2_block_mask_patch(model, block_length: int) -> None:
    """
    LLaDA2-MoE expects a 4D block-causal attention mask in forward, but in this codebase
    we patch the inner model to build that mask when attention_mask is None.
    This matches run_train.py / run_multi_train.py behavior.
    """
    import functools

    inner = getattr(model, "model", model)
    inner_cls = inner.__class__
    # This function is called for every batch (and for multiple block sizes).
    # We must (a) avoid re-wrapping forward (can cause infinite recursion), but
    # (b) still allow the active block length to change across calls.
    inner_cls._llada2_block_length = int(block_length)  # type: ignore[attr-defined]
    if getattr(inner_cls.forward, "_llada2_block_mask_patched", False):
        return

    orig_fwd = inner_cls.forward

    @functools.wraps(orig_fwd)
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
                return orig_fwd(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )
            bl = int(getattr(type(self), "_llada2_block_length", int(block_length)))
            n_blocks = (sl + bl - 1) // bl
            m = torch.tril(torch.ones(n_blocks, n_blocks, device=dev, dtype=torch.bool))
            m = (
                m.repeat_interleave(bl, dim=0)
                .repeat_interleave(bl, dim=1)[:sl, :sl]
                .unsqueeze(0)
                .unsqueeze(0)
            )
            # LLaDA2 expects attention_mask with shape (batch, 1, seq, seq).
            attention_mask = torch.where(m, 0.0, float("-inf")).to(torch.bfloat16)
            attention_mask = attention_mask.expand(bs, -1, -1, -1)
        return orig_fwd(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    # Mark wrapper so subsequent calls only update block length.
    _llada2_inner_forward._llada2_block_mask_patched = True  # type: ignore[attr-defined]
    inner_cls.forward = _llada2_inner_forward


@dataclass
class _RewardBundle:
    funcs: List[Any]
    max_sum: float


def _get_domain_rewards(domain: str) -> _RewardBundle:
    # Keep exactly the same mapping used by run_multi_train.py for these domains.
    # Package import so `python -m rl.block_r1` works without putting rl/ on PYTHONPATH.
    from rl.reward_func import (
        boxed_and_answer_tags_format_reward,
        correctness_reward_func,
        correctness_reward_func_math,
        countdown_reward_func,
        int_reward_func,
        knights_knaves_reward_func,
        sudoku_reward_func,
        xmlcount_reward_func,
        get_code_format_reward,
        code_reward,
    )

    domain_reward_map = {
        "gsm8k": [xmlcount_reward_func, int_reward_func, correctness_reward_func],
        "math": [correctness_reward_func_math, boxed_and_answer_tags_format_reward],
        "countdown": [countdown_reward_func],
        "sudoku": [sudoku_reward_func],
        "kodcode": [get_code_format_reward(language="python"), code_reward],
        "knights_and_knaves": [knights_knaves_reward_func],
    }
    if domain not in domain_reward_map:
        raise ValueError(f"Unsupported domain for Block-R1: {domain}")
    funcs = domain_reward_map[domain]
    # Assumption: each func returns roughly in [0,1]. Sum then normalize by count.
    max_sum = float(len(funcs)) if funcs else 1.0
    return _RewardBundle(funcs=funcs, max_sum=max_sum)


def _compute_reward(domain: str, prompt_msgs: Any, completion_text: str, example: Dict[str, Any]) -> Tuple[float, float]:
    bundle = _get_domain_rewards(domain)
    prompts = [prompt_msgs]
    completions = [[{"role": "assistant", "content": completion_text}]]
    # Avoid duplicate-keyword TypeError if a row ever contains prompts/completions keys.
    ex_kw = {k: v for k, v in example.items() if k not in ("prompts", "completions")}
    raw = 0.0
    for fn in bundle.funcs:
        # Reward functions accept extra kwargs (answer/target/solution/...); never drop them on fallback.
        try:
            r = fn(prompts=prompts, completions=completions, **ex_kw)
        except TypeError as e:
            # Only retry without `prompts=` when the function truly rejects that kwarg
            # (e.g. get_code_format_reward). Do not swallow TypeError raised inside fn bodies.
            msg = str(e)
            if "prompts" in msg and (
                "unexpected keyword" in msg or "got an unexpected keyword argument" in msg
            ):
                r = fn(completions=completions, **ex_kw)
            else:
                raise
        if r and r[0] is not None:
            raw += float(r[0])
    norm = raw / bundle.max_sum if bundle.max_sum > 0 else raw
    norm = float(max(0.0, min(1.0, norm)))
    return raw, norm


def _load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    revision: Optional[str],
    trust_remote_code: bool = True,
    use_4bit: bool = True,
):
    # Some LLaDA remote modeling files import `TransformersKwargs` from
    # `transformers.utils`, but certain transformers builds don't expose it.
    # Patch it in before any remote code import happens.
    try:
        from rl.llada2_compat import ensure_transformers_kwargs

        ensure_transformers_kwargs()
    except Exception:
        pass

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
    from transformers import BitsAndBytesConfig

    # Transformers may call caching_allocator_warmup with model._tp_plan is None (LLaDA remote code).
    # Same workaround as rl/run_train.py and rl/run_multi_train.py.
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

    try:
        cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, revision=revision
        )
    except ValueError:
        cfg = PretrainedConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, revision=revision
        )

    model_type = str(getattr(cfg, "model_type", "") or "")
    auto_map = getattr(cfg, "auto_map", {}) or {}

    load_kwargs: Dict[str, Any] = dict(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        revision=revision,
    )
    if model_type != "llada2_moe" and use_4bit:
        # Match the default pipeline behavior for big models.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if "AutoModelForCausalLM" in auto_map:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs).to(device)
    else:
        model = AutoModel.from_pretrained(**load_kwargs).to(device)

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    tok.pad_token = tok.eos_token
    model.config.use_cache = False
    return model, tok, model_type


@torch.no_grad()
def _generate_one_batch(
    model,
    tokenizer,
    model_type: str,
    prompt_ids: torch.Tensor,
    steps: int,
    gen_length: int,
    block_length: int,
):
    if model_type == "llada2_moe":
        from rl.llada2_compat import generate_llada2

        _apply_llada2_block_mask_patch(model, block_length=block_length)
        return generate_llada2(
            model=model,
            prompt=prompt_ids,
            tokenizer=tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=None,
        )
    else:
        from rl.eval.generate import generate

        return generate(
            model=model,
            prompt=prompt_ids,
            tokenizer=tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=None,
        )


def _infer_num_processes_for_resilient(args: argparse.Namespace) -> int:
    np = getattr(args, "num_processes", None)
    if np is not None and int(np) > 0:
        return int(np)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        return len([x for x in cvd.split(",") if x.strip() != ""])
    return 1


def _eval_multi_block_resilient_driver(args: argparse.Namespace) -> None:
    """
    Run each (domain, block_size) in a separate `accelerate launch` subprocess.

    This isolates CUDA OOM / hard crashes to a single slice so the outer driver can
    continue with the next block size (or domain) without exiting the whole job.

    Determinism: each slice uses the same --seed/--sample_n/--datasets row as the
    non-resilient path, so example_id / manifest order stay identical across runs.
    """
    domains = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    block_sizes = [int(x) for x in str(args.block_sizes).split(",") if str(x).strip()]
    sample_n = int(args.sample_n)
    gen_length = int(args.gen_length)
    steps = int(args.diffusion_steps)
    batch_size = int(args.batch_size)

    model_path = str(args.model_path)
    _PINNED_REVISIONS = {
        "inclusionAI/LLaDA2.1-mini": "bbb5715c881500b34234071e68dbf38c3d657c4e",
        "inclusionAI/LLaDA2.0-mini": "d23215abc5f5675daf171f6739d0386eab53f712",
    }
    revision = str(args.revision) if args.revision else _PINNED_REVISIONS.get(model_path)

    out_root = os.path.abspath(args.output_dir)
    safe_model = _safe_name(model_path.replace("/", "-"))
    out_dir = os.path.join(out_root, safe_model)
    tmp_dir = os.path.join(out_dir, "_tmp")
    _ensure_dir(tmp_dir)
    meta = {
        "model_path": model_path,
        "revision": revision,
        "datasets": domains,
        "block_sizes": block_sizes,
        "sample_n_per_domain": sample_n,
        "seed": int(args.seed),
        "gen_length": gen_length,
        "diffusion_steps": steps,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resilient": True,
        "note": "Each (domain, block_size) runs in a separate accelerate subprocess.",
    }
    _jsonl_write(os.path.join(out_dir, "META.jsonl"), [meta])

    num_proc = _infer_num_processes_for_resilient(args)
    print(
        f"[block_r1] resilient driver: {len(domains)} domains × {len(block_sizes)} slices; "
        f"num_processes={num_proc}",
        flush=True,
    )

    failed = 0
    ok = 0
    for domain in domains:
        for b in block_sizes:
            # Skip launching this slice if final merged file already exists.
            # This avoids wasting time loading model checkpoints only to skip later.
            final_path = os.path.join(out_root, safe_model, f"rewards_{domain}_b{int(b)}.jsonl")
            if os.path.exists(final_path):
                ok += 1
                print(
                    f"[block_r1] slice skip (exists): domain={domain} block={b} ({os.path.basename(final_path)})",
                    flush=True,
                )
                continue
            cmd = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                "--multi_gpu",
                "--num_processes",
                str(num_proc),
                "--num_machines",
                "1",
                "--mixed_precision",
                "no",
                "--dynamo_backend",
                "no",
                "--main_process_port",
                str(random.randint(20000, 30000)),
                "-m",
                "rl.block_r1",
                "eval_multi_block",
                "--model_path",
                model_path,
                "--datasets",
                domain,
                "--block_sizes",
                str(b),
                "--sample_n",
                str(sample_n),
                "--seed",
                str(int(args.seed)),
                "--gen_length",
                str(gen_length),
                "--diffusion_steps",
                str(steps),
                "--batch_size",
                str(batch_size),
                "--output_dir",
                out_root,
                "--only_domain",
                domain,
                "--only_block_size",
                str(int(b)),
            ]
            if args.revision:
                cmd.extend(["--revision", str(args.revision)])
            if bool(getattr(args, "no_4bit", False)):
                cmd.append("--no_4bit")

            print(f"[block_r1] slice start: domain={domain} block={b}", flush=True)
            p = subprocess.run(cmd, env=os.environ.copy())
            if p.returncode != 0:
                failed += 1
                print(
                    f"[block_r1] slice FAILED domain={domain} block={b} exit={p.returncode}; "
                    "continuing to next slice.",
                    flush=True,
                )
            else:
                ok += 1
                print(f"[block_r1] slice ok: domain={domain} block={b}", flush=True)

    print(
        f"[block_r1] resilient driver: done. ok_slices={ok} failed_slices={failed}",
        flush=True,
    )


def eval_multi_block(args: argparse.Namespace) -> None:
    if bool(getattr(args, "resilient", False)):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError(
                "eval_multi_block --resilient must be launched as a single process (no accelerate / torchrun). "
                "Example: CUDA_VISIBLE_DEVICES=0,1,2,3 python -m rl.block_r1 eval_multi_block --resilient ..."
            )
        _eval_multi_block_resilient_driver(args)
        return

    rank, world_size, local_rank = _init_dist()
    try:
        _maybe_set_seed(int(args.seed))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        # Optional safety: cap prompt tokens to prevent extremely long prompts from making
        # a single forward pass look "stuck" for a long time (and risking NCCL timeouts).
        # Use env var so scripts don't need updating: BLOCK_R1_MAX_PROMPT_TOKENS=2048 (etc.)
        _max_prompt_tokens_env = os.environ.get("BLOCK_R1_MAX_PROMPT_TOKENS", "").strip()
        max_prompt_tokens = int(_max_prompt_tokens_env) if _max_prompt_tokens_env else None
        # Heartbeat logging interval (batches). 0 disables.
        _hb_env = os.environ.get("BLOCK_R1_HEARTBEAT_EVERY", "").strip()
        heartbeat_every = int(_hb_env) if _hb_env else 50

        model_path = str(args.model_path)
        # Keep model code stable across time: if the caller doesn't pin a revision,
        # default to known-good commits (same policy as rl/run_train.py).
        _PINNED_REVISIONS = {
            "inclusionAI/LLaDA2.1-mini": "bbb5715c881500b34234071e68dbf38c3d657c4e",
            "inclusionAI/LLaDA2.0-mini": "d23215abc5f5675daf171f6739d0386eab53f712",
        }
        revision = str(args.revision) if args.revision else _PINNED_REVISIONS.get(model_path)
        domains = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
        block_sizes = [int(x) for x in str(args.block_sizes).split(",") if str(x).strip()]
        od = getattr(args, "only_domain", None)
        obs = getattr(args, "only_block_size", None)
        if (od is None) ^ (obs is None):
            raise ValueError("--only_domain and --only_block_size must be set together, or neither.")
        if od is not None:
            domains = [str(od)]
            block_sizes = [int(obs)]
        is_slice = obs is not None
        sample_n = int(args.sample_n)
        gen_length = int(args.gen_length)
        steps = int(args.diffusion_steps)
        batch_size = int(args.batch_size)

        out_root = os.path.abspath(args.output_dir)
        safe_model = _safe_name(model_path.replace("/", "-"))
        out_dir = os.path.join(out_root, safe_model)
        tmp_dir = os.path.join(out_dir, "_tmp")

        # Fast-path skip for slice mode: if final file already exists, do not load model.
        if is_slice and len(domains) == 1 and len(block_sizes) == 1:
            _final_path = os.path.join(out_dir, f"rewards_{domains[0]}_b{int(block_sizes[0])}.jsonl")
            if os.path.exists(_final_path):
                if _is_main(rank):
                    print(
                        f"[block_r1] domain={domains[0]} b={int(block_sizes[0])}: {os.path.basename(_final_path)} exists; skipping (no model load).",
                        flush=True,
                    )
                return

        if _is_main(rank):
            _ensure_dir(tmp_dir)
            # Resilient driver writes META.jsonl once; slice workers must not overwrite it.
            if not is_slice:
                meta = {
                    "model_path": model_path,
                    "revision": revision,
                    "datasets": domains,
                    "block_sizes": block_sizes,
                    "sample_n_per_domain": sample_n,
                    "seed": int(args.seed),
                    "gen_length": gen_length,
                    "diffusion_steps": steps,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "world_size": world_size,
                }
                _jsonl_write(os.path.join(out_dir, "META.jsonl"), [meta])
        _barrier()
        # Every rank must be able to write its part files.
        _ensure_dir(tmp_dir)

        model, tokenizer, model_type = _load_model_and_tokenizer(
            model_path=model_path,
            device=device,
            revision=revision,
            trust_remote_code=True,
            use_4bit=not bool(args.no_4bit),
        )
        model.eval()

        for domain in domains:
            ds = _load_train_dataset(domain)
            if len(ds) < sample_n and _is_main(rank):
                print(
                    f"[block_r1] domain={domain}: train len={len(ds)} < --sample_n={sample_n}; "
                    f"using {len(ds)} examples.",
                    flush=True,
                )
            ds = _subsample_fixed(ds, n=sample_n, seed=int(args.seed))
            n_domain = len(ds)

            # Deterministic per-domain IDs (same across models/runs with same seed).
            try:
                orig_idxs = ds["_orig_idx"]  # type: ignore[index]
                example_ids = [f"{domain}:{int(i):06d}" for i in orig_idxs]
            except Exception:
                example_ids = [f"{domain}:{i:06d}" for i in range(n_domain)]

            # Shard by global position to guarantee alignment across ranks.
            local_positions = [i for i in range(n_domain) if i % world_size == rank]

            if _is_main(rank):
                _ensure_dir(out_dir)
            _barrier()

            # Canonical IDs/order for this domain (same across block sizes and resilient slices).
            if _is_main(rank):
                manifest_path = os.path.join(out_dir, f"manifest_{domain}.jsonl")
                if not os.path.exists(manifest_path):
                    try:
                        orig_idxs = ds["_orig_idx"]  # type: ignore[index]
                    except Exception:
                        orig_idxs = None
                    _jsonl_write(
                        manifest_path,
                        [
                            {
                                "domain": domain,
                                "example_id": example_ids[i],
                                "pos": i,
                                **({"orig_idx": int(orig_idxs[i])} if orig_idxs is not None else {}),
                            }
                            for i in range(len(example_ids))
                        ],
                    )
            _barrier()

            for b in block_sizes:
                part_path = os.path.join(tmp_dir, f"{domain}.b{b}.rank{rank}.jsonl")
                final_path = os.path.join(out_dir, f"rewards_{domain}_b{b}.jsonl")

                # If final merged file already exists, skip this block size entirely.
                # This makes the job restart-friendly (e.g. after OOM / preemption).
                if os.path.exists(final_path):
                    if _is_main(rank):
                        print(
                            f"[block_r1] domain={domain} b={b}: {os.path.basename(final_path)} exists; skipping.",
                            flush=True,
                        )
                    _barrier()
                    continue

                # Resume support per-rank: if a part file exists, continue after the
                # already-written rows. Use unique ``pos`` coverage (not raw line count)
                # so duplicate lines from crashes/retries do not mis-align batches.
                start_offset = 0
                if os.path.exists(part_path):
                    try:
                        start_offset = _resume_index_in_local_positions(
                            part_path, local_positions
                        )
                    except Exception:
                        start_offset = 0

                # Mini-batch over this rank's positions (keep deterministic order)
                if _is_main(rank):
                    n_local = len(local_positions)
                    n_batches = (n_local + batch_size - 1) // batch_size if n_local else 0
                    print(
                        f"[block_r1] domain={domain} b={b}: start rank0 local_batches={n_batches} "
                        f"(batch_size={batch_size}, max_prompt_tokens={max_prompt_tokens or 'None'})",
                        flush=True,
                    )
                for start in range(start_offset, len(local_positions), batch_size):
                    if heartbeat_every > 0 and _is_main(rank) and start_offset == 0 and start > 0:
                        if (start // batch_size) % heartbeat_every == 0:
                            done = start
                            total = len(local_positions)
                            pct = (100.0 * done / total) if total else 100.0
                            print(
                                f"[block_r1] domain={domain} b={b}: rank0 progress {done}/{total} ({pct:.1f}%)",
                                flush=True,
                            )
                    chunk_pos = local_positions[start : start + batch_size]
                    examples = [ds[int(i)] for i in chunk_pos]
                    prompts_text = [_extract_prompt_text(ex, tokenizer) for ex in examples]
                    toks = _tokenize_prompts(
                        tokenizer, prompts_text, device=device, max_prompt_tokens=max_prompt_tokens
                    )
                    prompt_ids = toks["input_ids"]

                    out_ids = _generate_one_batch(
                        model=model,
                        tokenizer=tokenizer,
                        model_type=model_type,
                        prompt_ids=prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=b,
                    )
                    # Decode completion ONLY (exclude prompt). This is critical for correct rewards
                    # and for stable cross-model comparisons.
                    prompt_lens = [
                        int(x.sum().item()) for x in toks["attention_mask"].to("cpu")
                    ]

                    batch_rows: List[Dict[str, Any]] = []
                    for j, pos in enumerate(chunk_pos):
                        ex = examples[j]
                        pl = int(prompt_lens[j])
                        completion_ids = out_ids[j, pl:]
                        completion_text = tokenizer.decode(
                            completion_ids, skip_special_tokens=False
                        )
                        raw, norm = _compute_reward(domain, ex.get("prompt"), completion_text, ex)
                        batch_rows.append(
                            {
                                "domain": domain,
                                "example_id": example_ids[int(pos)],
                                "pos": int(pos),
                                "block_size": int(b),
                                "reward_raw": float(raw),
                                "reward_norm": float(norm),
                            }
                        )

                    # Stream-write per batch so partial results survive OOM/preemption.
                    _jsonl_append(part_path, batch_rows)

                    # Best-effort GPU memory hygiene to reduce long-run fragmentation.
                    del toks, prompt_ids, out_ids, examples, prompts_text, prompt_lens, batch_rows
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                _barrier()

                if _is_main(rank):
                    # Merge all ranks by global ``pos``. Deduplicate: same pos can appear
                    # twice in a part file after resume/restart; last line wins per pos.
                    part_paths = [
                        os.path.join(tmp_dir, f"{domain}.b{b}.rank{r}.jsonl")
                        for r in range(world_size)
                    ]
                    missing = [p for p in part_paths if not os.path.exists(p)]
                    if missing:
                        miss_msg = "\n  ".join(missing)
                        raise FileNotFoundError(
                            f"[block_r1] missing rank part files for merge (domain={domain} b={b} world_size={world_size}).\n"
                            f"  {miss_msg}\n"
                            "Likely a rank crashed/OOM'd before writing its part. Re-run this slice to regenerate.\n"
                        )
                    by_pos: Dict[int, Dict[str, Any]] = {}
                    for p in part_paths:
                        for row in _jsonl_iter(p):
                            by_pos[int(row["pos"])] = row

                    with open(final_path, "w", encoding="utf-8") as f:
                        for pos in sorted(by_pos):
                            f.write(json.dumps(by_pos[pos], ensure_ascii=False) + "\n")

                    # Clean tmp parts.
                    for r in range(world_size):
                        try:
                            os.remove(os.path.join(tmp_dir, f"{domain}.b{b}.rank{r}.jsonl"))
                        except OSError:
                            pass
                _barrier()

        if _is_main(rank):
            # Remove tmp dir if empty
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


    finally:
        _safe_destroy_process_group()


def build_block_r1(args: argparse.Namespace) -> None:
    rank, world_size, _local_rank = _init_dist()
    if world_size != 1:
        # This stage is pure file I/O; keep it single-process for determinism.
        if _is_main(rank):
            raise RuntimeError("build_block_r1 should be run with 1 process (no torchrun/accelerate).")
        return

    out_root = os.path.abspath(args.output_dir)
    model_a = _safe_name(str(args.model_a).replace("/", "-"))
    model_b = _safe_name(str(args.model_b).replace("/", "-"))
    domains = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    block_sizes = [int(x) for x in str(args.block_sizes).split(",") if str(x).strip()]

    require_a_gt_b = not bool(getattr(args, "no_filter_a_gt_b", False))
    multi_subdir = str(getattr(args, "multi_train_subdir", "block_r1_A_gt_B_multi_train"))
    train_path = os.path.join(out_root, multi_subdir, "train.jsonl")
    meta_path = os.path.join(out_root, multi_subdir, "META.json")
    _ensure_dir(os.path.dirname(train_path))

    resume = bool(getattr(args, "resume", False))
    multi_train_dir = os.path.join(out_root, multi_subdir)
    dedup_substring = bool(getattr(args, "dedup_substring", False))

    completed_domains: set = set()
    if resume:
        if os.path.exists(meta_path):
            prev_stats = json.load(open(meta_path, "r", encoding="utf-8"))
            # Safety: refuse to mix incompatible configs into a single output JSONL.
            if str(prev_stats.get("model_a")) != str(args.model_a) or str(prev_stats.get("model_b")) != str(args.model_b):
                raise RuntimeError(
                    "[block_r1] --resume detected existing META.json with different model_a/model_b.\n"
                    f"  existing: model_a={prev_stats.get('model_a')} model_b={prev_stats.get('model_b')}\n"
                    f"  current:  model_a={args.model_a} model_b={args.model_b}\n"
                    f"  meta_path={meta_path}"
                )
            if list(prev_stats.get("block_sizes", [])) != block_sizes:
                raise RuntimeError(
                    "[block_r1] --resume detected existing META.json with different block_sizes.\n"
                    f"  existing: {prev_stats.get('block_sizes')}\n"
                    f"  current:  {block_sizes}\n"
                    f"  meta_path={meta_path}"
                )
            if bool(prev_stats.get("require_a_gt_b")) != require_a_gt_b:
                raise RuntimeError(
                    "[block_r1] --resume detected existing META.json with different require_a_gt_b.\n"
                    f"  existing: {prev_stats.get('require_a_gt_b')}\n"
                    f"  current:  {require_a_gt_b}\n"
                    f"  meta_path={meta_path}"
                )
            completed_domains = set((prev_stats.get("per_domain") or {}).keys())
        elif os.path.exists(train_path):
            # Fallback: META.json missing; infer completed domains by scanning train.jsonl.
            #
            # Best-effort only. The domain-atomic append logic below makes this more reliable by
            # ensuring a domain appears in train.jsonl only after that domain fully finishes.
            for row in _jsonl_iter(train_path):
                d = row.get("br1_domain")
                if isinstance(d, str) and d:
                    completed_domains.add(d)

    stats: Dict[str, Any]
    if resume and os.path.exists(meta_path):
        # Start from previous stats and extend.
        stats = json.load(open(meta_path, "r", encoding="utf-8"))
        stats.setdefault("per_domain", {})
        stats.setdefault("total_kept", 0)
        stats.setdefault("total_seen", 0)
        # Keep a record of the requested domain list too (useful for later provenance).
        prev_requested = list(stats.get("datasets_requested", stats.get("datasets", [])) or [])
        stats["datasets_requested"] = prev_requested + [d for d in domains if d not in prev_requested]
        # Keep datasets as "completed domains" (union), for backwards compatibility with older readers.
        prev_completed = list(stats.get("datasets", []) or [])
        union_completed = prev_completed + [d for d in sorted(completed_domains) if d not in prev_completed]
        stats["datasets"] = union_completed
        stats["multi_train_dir"] = multi_train_dir
    else:
        stats = {
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "datasets": [],
            "datasets_requested": domains,
            "block_sizes": block_sizes,
            "require_a_gt_b": require_a_gt_b,
            "multi_train_dir": multi_train_dir,
            "per_domain": {},
            "total_kept": 0,
            "total_seen": 0,
            "note": (
                "Each line: original train columns (e.g. prompt) plus br1_* metadata: "
                "br1_best_block_size, br1_reward_norm_a/b, br1_delta_a_minus_b, br1_per_block_reward_*. "
                "Load: datasets.load_dataset('json', data_files=train.jsonl, split='train')."
            ),
        }

    # Stream one JSONL for all domains; do not hold all examples in RAM.
    # For resume safety, we append output atomically per domain:
    # write domain rows to a tmp file first, then append to train.jsonl only when that domain finishes.
    if not (resume and os.path.exists(train_path)):
        # Fresh build: truncate to avoid mixing with older outputs.
        with open(train_path, "w", encoding="utf-8"):
            pass

    for domain in domains:
        if resume and domain in completed_domains:
            continue

        man_a = os.path.join(out_root, model_a, f"manifest_{domain}.jsonl")
        man_b = os.path.join(out_root, model_b, f"manifest_{domain}.jsonl")
        if not os.path.exists(man_a) or not os.path.exists(man_b):
            raise FileNotFoundError(
                f"Missing manifest for {domain}. Expected:\n  {man_a}\n  {man_b}"
            )
        manifest = _jsonl_read(man_a)

        ra = _load_reward_norm_map(out_root, model_a, domain, block_sizes)
        rb = _load_reward_norm_map(out_root, model_b, domain, block_sizes)

        # Full train split; index by orig_idx from manifest (pre-shuffle index in block_r1 eval).
        train_ds = _load_train_dataset(domain)
        n_train = len(train_ds)

        kept = 0
        seen = len(manifest)
        skipped_no_idx = 0
        skipped_not_a_gt_b = 0
        skipped_missing_reward = 0

        tmp_domain_path = os.path.join(multi_train_dir, f"train.{domain}.tmp.jsonl")
        with open(tmp_domain_path, "w", encoding="utf-8") as tmp_f:
            for mrow in manifest:
                ex_id = str(mrow["example_id"])
                orig_idx = mrow.get("orig_idx")
                if orig_idx is None:
                    skipped_no_idx += 1
                    continue
                oi = int(orig_idx)
                if oi < 0 or oi >= n_train:
                    skipped_no_idx += 1
                    continue

                best_b: Optional[int] = None
                best_delta = -float("inf")
                per_av: Dict[int, float] = {}
                per_bv: Dict[int, float] = {}
                missing = False
                for b in block_sizes:
                    ka = (ex_id, int(b))
                    a = ra.get(ka)
                    bb = rb.get(ka)
                    if a is None or bb is None:
                        missing = True
                        break
                    da = float(a)
                    db = float(bb)
                    per_av[int(b)] = da
                    per_bv[int(b)] = db
                    adv = da - db
                    if adv > best_delta:
                        best_delta = adv
                        best_b = int(b)
                if missing or best_b is None:
                    skipped_missing_reward += 1
                    continue

                if require_a_gt_b and not (best_delta >= 0):
                    skipped_not_a_gt_b += 1
                    continue

                ex = train_ds[oi]
                rec = _train_row_to_record(dict(ex))
                out = {
                    **rec,
                    "br1_domain": domain,
                    "br1_example_id": ex_id,
                    "br1_pos": int(mrow.get("pos", -1)),
                    "br1_orig_idx": oi,
                    "br1_best_block_size": best_b,
                    "br1_reward_norm_a": per_av[best_b],
                    "br1_reward_norm_b": per_bv[best_b],
                    "br1_delta_a_minus_b": best_delta,
                    "br1_per_block_reward_a": per_av,
                    "br1_per_block_reward_b": per_bv,
                }
                tmp_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1

        with open(train_path, "a", encoding="utf-8") as train_f, open(tmp_domain_path, "r", encoding="utf-8") as tmp_f:
            for line in tmp_f:
                train_f.write(line)
        try:
            os.remove(tmp_domain_path)
        except OSError:
            pass

        stats["per_domain"][domain] = {
            "seen_manifest": seen,
            "kept": kept,
            "skipped_no_orig_idx": skipped_no_idx,
            "skipped_missing_reward": skipped_missing_reward,
            "skipped_not_a_gt_b": skipped_not_a_gt_b,
        }
        stats["total_kept"] += kept
        stats["total_seen"] += seen
        if domain not in (stats.get("datasets") or []):
            stats["datasets"] = list(stats.get("datasets") or []) + [domain]

        # Persist META per-domain so resume can skip reliably.
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        if dedup_substring:
            ded = _dedup_substring_inplace_jsonl(train_path)
            stats.setdefault("dedup", {})
            stats["dedup"]["strategy"] = "substring_remove_shorter"
            stats["dedup"]["last_run"] = {"after_domain": domain, **ded}
            stats["dedup"]["total_removed"] = int(stats["dedup"].get("total_removed", 0)) + int(ded.get("removed", 0))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

        del train_ds

    # Write once more at end for good measure.
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rl.block_r1")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval_multi_block", help="Stage-2: eval train split under multiple block sizes")
    p_eval.add_argument("--model_path", required=True)
    p_eval.add_argument("--revision", default=None)
    p_eval.add_argument(
        "--datasets",
        required=True,
        help='Comma-separated: "gsm8k,math,countdown,sudoku,kodcode,knights_and_knaves"',
    )
    p_eval.add_argument("--block_sizes", required=True, help='Comma-separated: "4,8,16,32,64,128"')
    p_eval.add_argument("--sample_n", type=int, default=7500)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--gen_length", type=int, default=512)
    p_eval.add_argument("--diffusion_steps", type=int, default=256)
    p_eval.add_argument("--batch_size", type=int, default=8)
    p_eval.add_argument("--output_dir", required=True)
    p_eval.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading for non-llada2 models")
    p_eval.add_argument(
        "--resilient",
        action="store_true",
        help=(
            "Run each (domain, block_size) in a separate accelerate subprocess so OOM/kill only affects "
            "that slice; the driver continues with the next slice. "
            "Must be invoked WITHOUT accelerate (single-process driver). "
            "Prefer updating block_data.sh to call `python -m rl.block_r1 eval_multi_block --resilient ...`."
        ),
    )
    p_eval.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="For --resilient: GPU count passed to `accelerate launch` (default: infer from CUDA_VISIBLE_DEVICES).",
    )
    p_eval.add_argument("--only_domain", default=None, help=argparse.SUPPRESS)
    p_eval.add_argument("--only_block_size", type=int, default=None, help=argparse.SUPPRESS)
    p_eval.set_defaults(func=eval_multi_block)

    p_build = sub.add_parser(
        "build_block_r1",
        help=(
            "Build multi-domain training JSONL: per example, argmax block for (reward_A - reward_B); "
            "by default keep only where A>B at that block. Streams train.jsonl (no GPU; bounded RAM)."
        ),
    )
    p_build.add_argument("--model_a", required=True, help="Model A (e.g. inclusionAI/LLaDA2.0-mini)")
    p_build.add_argument("--model_b", required=True, help="Model B (e.g. GSAI-ML/LLaDA-8B-Instruct)")
    p_build.add_argument("--datasets", required=True)
    p_build.add_argument("--block_sizes", required=True)
    p_build.add_argument("--output_dir", required=True)
    p_build.add_argument(
        "--multi_train_subdir",
        default="block_r1_A_gt_B_multi_train",
        help="Under output_dir, write train.jsonl and META.json in this subdirectory.",
    )
    p_build.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append mode: if train.jsonl/META.json already exist, skip domains already present in META.json "
            "(or infer from train.jsonl if META is missing), and append only new domains."
        ),
    )
    p_build.add_argument(
        "--dedup_substring",
        action="store_true",
        help=(
            "After each exported domain, run a consistent dedup pass across train.jsonl: "
            "if one question/prompt is a substring of another, remove the shorter sample."
        ),
    )
    p_build.add_argument(
        "--no_filter_a_gt_b",
        action="store_true",
        help="Also export examples where A is not strictly better than B at the chosen block (argmax A-B).",
    )
    p_build.set_defaults(func=build_block_r1)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

