"""
Block-R1 offline dataset training helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from datasets import Dataset


def snap_block_size_for_gen_length(
    block_size: int, gen_length: int, *, fallback: int = 32
) -> int:
    """
    Diffusion generation requires ``gen_length % block_size == 0``.
    If the stored block is invalid, pick the largest divisor of ``gen_length``
    not exceeding ``block_size`` (else ``fallback`` snapped the same way).
    """
    gl = int(gen_length)
    if gl <= 0:
        return max(1, int(fallback))
    b = int(block_size)
    if b <= 0:
        b = int(fallback)
    b = min(b, gl)
    if gl % b == 0:
        return b
    for d in range(b, 0, -1):
        if gl % d == 0:
            return d
    fb = max(1, min(int(fallback), gl))
    for d in range(fb, 0, -1):
        if gl % d == 0:
            return d
    return 1


def load_block_r1_train_jsonl(path: str, seed: int = 42) -> Dataset:
    """
    Load ``train.jsonl`` and ensure a consistent schema across rows.

    HF Datasets' JSON loader can fail with ``DatasetGenerationCastError`` when JSONL rows
    introduce new columns partway through the file (multi-domain data often does). We
    therefore parse JSONL ourselves, normalize keys, and build the Dataset from the full
    list with a unified set of columns.
    """

    import json

    _STRUCT_KEYS = {"br1_per_block_reward_a", "br1_per_block_reward_b"}
    _LIST_KEYS = {"numbers", "nums", "test_list"}

    def _normalize_prompt(v: Any):
        if v is None:
            return None
        # expected: List[{"role": str, "content": str}]
        if isinstance(v, str):
            s = v.strip()
            return [{"role": "user", "content": s}] if s else []
        if isinstance(v, dict):
            # single message dict -> wrap
            if "role" in v and "content" in v:
                return [{"role": str(v.get("role")), "content": str(v.get("content"))}]
            return [{"role": "user", "content": json.dumps(v, ensure_ascii=False)}]
        if isinstance(v, list):
            out = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, dict) and "role" in item and "content" in item:
                    out.append(
                        {"role": str(item.get("role")), "content": str(item.get("content"))}
                    )
                elif isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append({"role": "user", "content": s})
                else:
                    out.append(
                        {"role": "user", "content": json.dumps(item, ensure_ascii=False)}
                    )
            return out
        # fallback: serialize
        return [{"role": "user", "content": json.dumps(v, ensure_ascii=False)}]

    def _normalize_value(k: str, v: Any):
        if v is None:
            return None
        if k == "prompt":
            return _normalize_prompt(v)
        if k in _LIST_KEYS:
            if isinstance(v, list):
                return v
            return [v]
        if k in _STRUCT_KEYS:
            # expect dict-like; if not, serialize into dict wrapper to keep type stable
            if isinstance(v, dict):
                return v
            return {"_value": json.dumps(v, ensure_ascii=False)}

        # For all other keys: keep scalars; serialize list/dict to JSON string so
        # pyarrow doesn't see mixed container/scalar types within one column.
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    rows: List[Dict[str, Any]] = []
    all_keys = set()
    with open(path, "r") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                ex = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"[Block-R1] Invalid JSON at line {ln} in {path}") from e

            # Normalize common cross-domain mismatches so downstream reward routing works.
            # - Some sources use 'problem' instead of 'question'
            if (ex.get("question") is None or str(ex.get("question") or "").strip() == "") and (
                ex.get("problem") is not None
            ):
                ex["question"] = ex.get("problem")
            # - Some sources use 'nums' instead of 'numbers' (countdown)
            if ex.get("numbers") is None and ex.get("nums") is not None:
                ex["numbers"] = ex.get("nums")

            # Ensure domain exists (reward routing uses 'domain')
            d = ex.get("domain")
            if d is None or (isinstance(d, str) and not str(d).strip()):
                ex["domain"] = ex.get("br1_domain") or "general"

            # Normalize container/scalar types to avoid ArrowInvalid.
            ex = {k: _normalize_value(k, v) for k, v in ex.items()}

            rows.append(ex)
            all_keys.update(ex.keys())

    if not rows:
        return Dataset.from_list([])

    # Make schema consistent: add missing keys with None.
    keys = sorted(all_keys)
    norm_rows: List[Dict[str, Any]] = []
    for ex in rows:
        r = {k: ex.get(k, None) for k in keys}
        norm_rows.append(r)

    ds = Dataset.from_list(norm_rows)
    ds = ds.shuffle(seed=seed)
    return ds


def infer_block_r1_domains(ds: Dataset) -> List[str]:
    """Unique ``br1_domain`` (or ``domain``) values present in the dataset."""
    col = "br1_domain" if "br1_domain" in ds.column_names else "domain"
    if col not in ds.column_names:
        return ["general"]
    seen = set()
    for v in ds[col]:
        s = str(v or "").strip() or "general"
        seen.add(s)
    return sorted(seen)


# Columns required so ``create_multi_domain_reward_func`` + DOMAIN_REWARD_MAP can score
# rows the same way as ``run_multi_train.create_multi_domain_dataset`` (see rl/reward_func.py).
_BLOCK_R1_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "gsm8k": ("prompt", "answer"),
    "math": ("prompt", "answer"),
    "countdown": ("prompt", "target", "numbers"),
    "sudoku": ("prompt", "puzzle", "solution"),
    "kodcode": ("prompt", "test_list"),
    "knights_and_knaves": ("prompt", "answer"),
}


def validate_block_r1_reward_schema(ds: Dataset, *, domains: List[str]) -> None:
    """
    Fail fast if any domain is missing columns needed by its reward functions.

    ``build_block_r1`` should have copied full loader rows via ``_train_row_to_record``;
    this catches truncated JSONL or schema drift.
    """
    if len(ds) == 0:
        raise ValueError("[Block-R1] empty dataset after load/filter.")
    names = set(ds.column_names)
    dom_col = "domain" if "domain" in names else "br1_domain"
    if dom_col not in names:
        raise ValueError("[Block-R1] expected 'domain' or 'br1_domain' column.")

    for dom in domains:
        req = _BLOCK_R1_REQUIRED_COLUMNS.get(dom)
        if not req:
            continue
        missing_cols = [c for c in req if c not in names]
        if missing_cols:
            raise ValueError(
                f"[Block-R1] Dataset schema missing columns {missing_cols} for domain={dom!r}. "
                f"Present columns: {sorted(names)}"
            )
        # One row check: first example of this domain must have non-None label fields
        idx0 = None
        for i in range(len(ds)):
            if str(ds[i][dom_col] or "").strip() == dom:
                idx0 = i
                break
        if idx0 is None:
            continue
        row = ds[idx0]
        bad = []
        for c in req:
            v = row.get(c)
            if v is None:
                bad.append(c)
        if bad:
            raise ValueError(
                f"[Block-R1] domain={dom!r}: first row has null label fields {bad}. "
                "Rebuild train.jsonl or check _train_row_to_record / JSON export."
            )


def block_sizes_for_inputs(
    inputs: Sequence[dict],
    *,
    gen_length: int,
    fallback_block: int,
) -> List[int]:
    """Per-row snapped block sizes aligned with ``inputs`` order."""
    out: List[int] = []
    for x in inputs:
        raw = x.get("br1_best_block_size", fallback_block)
        try:
            b = int(raw)
        except (TypeError, ValueError):
            b = int(fallback_block)
        out.append(snap_block_size_for_gen_length(b, gen_length, fallback=fallback_block))
    return out
