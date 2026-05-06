"""
Load LLM360/guru-RL-92k for R1 multi-domain training.

Dataset card: https://huggingface.co/datasets/LLM360/guru-RL-92k
Hub ``train/*.parquet`` shards use mismatched column layouts; this module loads each file
with ``Dataset.from_parquet`` and concatenates after mapping (see ``_load_guru_via_shard_parquet``).
Mapped rows use Arrow ``large_string`` / explicit ``Features`` so very long prompts or tests do not
hit PyArrow ``list<string>`` offset overflow. ``prompt`` and ``guru_extra_info`` are stored as JSON
strings and decoded to chat lists / dicts via ``set_transform`` (batched) for TRL.
Rows are converted to the conversational `prompt` format expected by dLLM-R1 trainers.

Reward (training): `reward_func.guru_unified_reward_func` uses the vendored router
`rl.eval.guru.default_compute_score` (Reasoning360 `verl/utils/reward_score` logic, trimmed
to Guru-relevant handlers only). Same fields as `NaiveRewardManager`: `data_source`,
`reward_model["ground_truth"]`, and `extra_info`. Install deps from Reasoning360 README
if any import fails.

`guru_skill` is a coarse heuristic used for **R1 block-size routing** (separate centroids / Q
per bucket). **Scoring** still follows `guru_data_source` → `rl.eval.guru`, not `guru_skill` alone.

R1 ``domain`` column on Guru rows is ``guru_math`` / ``guru_code`` / … (see
``GURU_R1_DOMAIN_KEYS``), not the string ``guru``. Eval benchmarks (gsm8k, …) are not in
that list; ``select_block_size_by_embedding`` routes by prompt similarity to these centroids.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download

GURU_HF_ID = "LLM360/guru-RL-92k"

# Avoid PyArrow "offset overflow" when concatenating huge strings in list/struct columns
# (default ``string`` uses int32 offsets; ``large_string`` uses int64).
GURU_TRAIN_RECORD_FEATURES = Features(
    {
        # JSON list of {"role","content"} — avoids Arrow list<struct<string>> offset overflow
        # and matches TRL (Arrow would decode struct-of-lists, breaking chat templates).
        "prompt": Value("large_string"),
        "domain": Value("string"),
        "guru_skill": Value("string"),
        "test_list": Sequence(Value("large_string")),
        "guru_entry_point": Value("large_string"),
        "guru_math_answer": Value("large_string"),
        "guru_mc_answer": Value("large_string"),
        "guru_task_id": Value("large_string"),
        "guru_data_source": Value("large_string"),
        "guru_reward_ground_truth": Value("large_string"),
        "guru_extra_info": Value("large_string"),
        "guru_code_prefix": Value("large_string"),
    }
)

# R1 BlockSizeController domain names for Guru (one centroid + Q per bucket).
GURU_R1_DOMAIN_KEYS: Tuple[str, ...] = (
    "guru_math",
    "guru_code",
    "guru_mc",
    "guru_table",
    "guru_logic",
    "guru_other",
)

_GURU_SKILL_TO_R1: Dict[str, str] = {
    "math": "guru_math",
    "code": "guru_code",
    "mc": "guru_mc",
    "table": "guru_table",
    "logic": "guru_logic",
    "other": "guru_other",
}


def guru_r1_domain_from_skill(skill: str) -> str:
    """Map coarse Guru skill → R1 bandit domain (must be in ``GURU_R1_DOMAIN_KEYS``)."""
    s = (skill or "other").strip().lower()
    return _GURU_SKILL_TO_R1.get(s, "guru_other")


def _coerce_mapping(val: Any) -> Optional[Dict[str, Any]]:
    """HF/Parquet may store dict columns as JSON strings."""
    if val is None:
        return None
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, str) and val.strip():
        try:
            obj = json.loads(val)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _infer_skill(ability: Optional[str], data_source: Optional[str]) -> str:
    a = (ability or "").lower().strip()
    ds = (data_source or "").lower().strip()
    if a == "codegen" or any(
        x in ds
        for x in (
            "leetcode",
            "taco",
            "livecode",
            "primeintellect",
            "codegen",
            "lcb",
        )
    ):
        return "code"
    if a == "math" or any(
        x in ds for x in ("math", "dapo", "deepscaler", "or1", "skywork")
    ):
        return "math"
    if any(
        x in ds
        for x in (
            "mmlu",
            "hellaswag",
            "arc",
            "gpqa",
            "multiple_choice",
            "multiplechoice",
        )
    ):
        return "mc"
    if any(x in ds for x in ("hitab", "multihiertt", "table", "tabular")):
        return "table"
    if any(x in ds for x in ("arc-agi", "barc", "puzzle", "logic", "zebra")):
        return "logic"
    return "other"


def _parse_mc_gt(row: Dict[str, Any]) -> str:
    """Best-effort ground-truth letter for MC-style rows."""
    io = row.get("input_output")
    if isinstance(io, str) and io.strip():
        try:
            io = json.loads(io)
        except json.JSONDecodeError:
            io = None
    if isinstance(io, list) and io:
        last = io[-1]
        if isinstance(last, dict):
            o = last.get("output", "")
            if isinstance(o, str) and len(o.strip()) == 1:
                return o.strip().upper()
    rm = _coerce_mapping(row.get("reward_model"))
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, str):
            try:
                obj = json.loads(gt)
                if isinstance(obj, dict):
                    for k in ("answer", "letter", "choice"):
                        v = obj.get(k)
                        if isinstance(v, str) and len(v.strip()) == 1:
                            return v.strip().upper()
            except json.JSONDecodeError:
                pass
    return ""


def _normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    if isinstance(prompt, list):
        out = []
        for m in prompt:
            if isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": str(m["role"]), "content": str(m["content"])})
        return out if out else [{"role": "user", "content": str(prompt)}]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": str(prompt)}]


def _reward_ground_truth_str(row: Dict[str, Any]) -> str:
    """String passed to Reasoning360 `default_compute_score(..., ground_truth=...)`."""
    rm = _coerce_mapping(row.get("reward_model"))
    gt: Any = None
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
    if gt is None:
        return ""
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False)
    if isinstance(gt, bytes):
        return gt.decode("utf-8", errors="replace")
    return str(gt)


def _extra_info_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    ex = _coerce_mapping(row.get("extra_info"))
    if ex is not None:
        return dict(ex)
    if isinstance(row.get("extra_info"), dict):
        return dict(row["extra_info"])
    return {}


def _decode_guru_arrow_row_for_trainer(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    ``Dataset.set_transform`` passes a **batch** (dict of columns as lists), not one row.
    Decode JSON ``prompt`` / ``guru_extra_info`` for TRL chat templates and reward kwargs.

    Batches may omit ``prompt`` (e.g. ``dataset["guru_data_source"]`` column-only reads);
    only decode keys that are present.
    """
    out = dict(batch)
    if "prompt" in batch:
        decoded_prompts: List[Any] = []
        for p in batch["prompt"]:
            if isinstance(p, str):
                try:
                    loaded = json.loads(p)
                    decoded_prompts.append(
                        loaded
                        if isinstance(loaded, list)
                        else [{"role": "user", "content": str(loaded)}]
                    )
                except json.JSONDecodeError:
                    decoded_prompts.append([{"role": "user", "content": p}])
            else:
                decoded_prompts.append(p)
        out["prompt"] = decoded_prompts

    if "guru_extra_info" in batch:
        decoded_extras: List[Any] = []
        for exs in batch["guru_extra_info"]:
            if isinstance(exs, str) and exs.strip():
                try:
                    parsed = json.loads(exs)
                    decoded_extras.append(parsed if isinstance(parsed, dict) else {})
                except json.JSONDecodeError:
                    decoded_extras.append({})
            elif isinstance(exs, dict):
                decoded_extras.append(exs)
            else:
                decoded_extras.append({})
        out["guru_extra_info"] = decoded_extras
    return out


def _coerce_test_field(val: Any) -> str:
    """Parquet shards use ``test`` or ``tests``; values may be str or structured."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False).strip()
    return str(val).strip()


def _row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    skill = _infer_skill(row.get("ability"), row.get("data_source"))
    test = _coerce_test_field(row.get("test") or row.get("tests"))
    if skill == "code" and not test:
        skill = "other"

    response = row.get("response") or row.get("completion") or ""
    if isinstance(response, bytes):
        response = response.decode("utf-8", errors="replace")
    response = str(response)

    mc_gt = _parse_mc_gt(row) if skill == "mc" else ""

    test_list: List[str] = [test] if skill == "code" and test else []

    extra = dict(_extra_info_dict(row))
    prompt_msgs = _normalize_prompt(row.get("prompt"))
    if not any(str(m.get("content", "")).strip() for m in prompt_msgs) and row.get(
        "problem"
    ) is not None:
        prob = row["problem"]
        if isinstance(prob, bytes):
            prob = prob.decode("utf-8", errors="replace")
        prompt_msgs = [{"role": "user", "content": str(prob)}]

    md = _coerce_mapping(row.get("metadata"))
    entry_point = str(row.get("entry_point") or "")
    if not entry_point and isinstance(md, dict) and md.get("func_name"):
        entry_point = str(md["func_name"])

    q_existing = extra.get("question")
    if not (isinstance(q_existing, str) and q_existing.strip()):
        for m in reversed(prompt_msgs):
            role = str(m.get("role", "")).lower()
            if role in ("user", "human"):
                extra["question"] = str(m.get("content", ""))
                break

    prefix = ""
    p = extra.get("prefix")
    if isinstance(p, str):
        prefix = p
    elif p is not None:
        prefix = str(p)
    if not prefix.strip() and row.get("starter_code") is not None:
        sc = row["starter_code"]
        if isinstance(sc, bytes):
            sc = sc.decode("utf-8", errors="replace")
        prefix = str(sc)

    extra_json = json.dumps(extra, ensure_ascii=False, default=str) if extra else "{}"
    prompt_json = json.dumps(prompt_msgs, ensure_ascii=False)

    return {
        "prompt": prompt_json,
        "domain": guru_r1_domain_from_skill(skill),
        "guru_skill": skill,
        "test_list": test_list,
        "guru_entry_point": entry_point,
        "guru_math_answer": response,
        "guru_mc_answer": mc_gt,
        "guru_task_id": str(row.get("task_id") or ""),
        # Reasoning360 routing key (exact HF string, e.g. math__..., codegen__..., table__...)
        "guru_data_source": str(row.get("data_source") or ""),
        "guru_reward_ground_truth": _reward_ground_truth_str(row),
        "guru_extra_info": extra_json,
        "guru_code_prefix": prefix,
    }


def _load_guru_via_shard_parquet(hf_id: str, split: str) -> Dataset:
    """
    HF ``load_dataset`` merges all ``train/*.parquet`` into one Arrow table; Guru shards use
    incompatible schemas (e.g. codegen LeetCode vs LiveCodeBench columns), which raises
    CastError. Load each file with ``Dataset.from_parquet``, map to a common row dict, then
    ``concatenate_datasets``.
    """
    repo_dir = snapshot_download(
        repo_id=hf_id,
        repo_type="dataset",
        allow_patterns=[f"{split}/*.parquet"],
    )
    root = Path(repo_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(f"guru-RL-92k: missing split directory {root}")
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"guru-RL-92k: no parquet files under {root}")
    pieces: List[Dataset] = []
    for path in paths:
        shard = Dataset.from_parquet(str(path))
        pieces.append(
            shard.map(
                lambda ex: _row_to_record(ex),
                remove_columns=shard.column_names,
                features=GURU_TRAIN_RECORD_FEATURES,
                writer_batch_size=16,
                desc=f"guru shard {path.name}",
            )
        )
    return concatenate_datasets(pieces)


def load_guru_rl_train(
    hf_id: str = GURU_HF_ID,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load Guru RL parquet data and map to dLLM-R1 row schema.

    The canonical Hub repo ``LLM360/guru-RL-92k`` mixes parquet shards with incompatible
    schemas; ``datasets.load_dataset`` fails while merging. That id is loaded per-shard
    (see ``_load_guru_via_shard_parquet``). Other ``hf_id`` values use unified load, with
    a fallback to per-shard load if merge still fails.
    """
    if hf_id == GURU_HF_ID:
        ds = _load_guru_via_shard_parquet(hf_id, split)
        if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))
    else:
        try:
            ds = load_dataset(hf_id, split=split)
            if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
                ds = ds.shuffle(seed=seed).select(range(max_samples))
            ds = ds.map(
                lambda ex: _row_to_record(ex),
                remove_columns=ds.column_names,
                features=GURU_TRAIN_RECORD_FEATURES,
                writer_batch_size=16,
                desc="Mapping guru-RL-92k rows",
            )
        except Exception as e:
            warnings.warn(
                f"guru dataset {hf_id!r}: unified load failed ({type(e).__name__}: {e}); "
                f"loading parquet shards separately.",
                stacklevel=2,
            )
            ds = _load_guru_via_shard_parquet(hf_id, split)
            if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
                ds = ds.shuffle(seed=seed).select(range(max_samples))
    # Rows without data_source cannot be scored by rl.eval.guru (would yield silent 0s in reward).
    n_before = len(ds)
    ds = ds.filter(lambda x: bool(str(x.get("guru_data_source") or "").strip()))
    if len(ds) < n_before and n_before > 0:
        warnings.warn(
            f"guru-RL-92k: dropped {n_before - len(ds)}/{n_before} rows with empty data_source.",
            stacklevel=2,
        )
    ds.set_transform(_decode_guru_arrow_row_for_trainer)
    return ds
