#!/usr/bin/env python3
"""Copy Guru-relevant modules from a Reasoning360 checkout into rl/eval/guru/.

Does not overwrite ``rl/eval/guru/__init__.py`` (keep the trimmed router in dLLM-R1).

Usage (from dLLM-R1 repo root)::

    REASONING360_ROOT=/path/to/Reasoning360 python rl/eval/guru/scripts/vendor_guru_reward.py

Requires Python 3.9+.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Paths under verl/utils/reward_score/ to vendor (Guru / multi-domain subset only).
VENDOR_REL_PATHS = (
    "arcagi.py",
    "codeio.py",
    "coder1",
    "cruxeval",
    "gpqa.py",
    "graph_dataset.py",
    "math_dapo.py",
    "math_llm_judge",
    "math.py",
    "naive_dapo.py",
    "prime_math",
    "puzzles_dataset.py",
    "py_functional.py",
    "stem_llm_judge",
    "supergpqa.py",
    "tablereason.py",
    "zebra_puzzle.py",
)


def main() -> int:
    root = os.environ.get("REASONING360_ROOT")
    if not root:
        print("Set REASONING360_ROOT to your Reasoning360 repository root.", file=sys.stderr)
        return 1
    src_base = Path(root) / "verl" / "utils" / "reward_score"
    if not src_base.is_dir():
        print(f"Missing reward_score dir: {src_base}", file=sys.stderr)
        return 1

    def _repo_root() -> Path:
        p = Path(__file__).resolve()
        for ancestor in p.parents:
            if (ancestor / "rl" / "eval" / "guru" / "__init__.py").is_file():
                return ancestor
        raise RuntimeError(
            "Cannot find repo root (expected rl/eval/guru/__init__.py on a parent path)."
        )

    repo_root = _repo_root()
    dst_base = repo_root / "rl" / "eval" / "guru"
    dst_base.mkdir(parents=True, exist_ok=True)

    for rel in VENDOR_REL_PATHS:
        s = src_base / rel
        d = dst_base / rel
        if not s.exists():
            print(f"Skip (not in upstream): {rel}", file=sys.stderr)
            continue
        if s.is_dir():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

    print(f"Updated {dst_base} from {src_base}")
    print("Keep rl/eval/guru/__init__.py as the trimmed default_compute_score router.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
