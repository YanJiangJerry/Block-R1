"""
Universal push-to-hub uploader for this repository.

It auto-discovers all methods/models/checkpoints under the checkpoints root and
builds normalized HF repo names from real folder paths.

Commented usage examples (from repo root, PYTHONPATH includes project root if needed):

# 1) Dry-run all discovered targets (recommended first)
# python -m rl.eval.push_to_hub --dry-run

# 2) Upload everything under checkpoints/ to diffusion-reasoning/*
# python -m rl.eval.push_to_hub --namespace diffusion-reasoning

# 3) Upload only wd1 and b1_wd1 methods, latest checkpoint for each model
# python -m rl.eval.push_to_hub --methods wd1,wd1_math,b1_wd1_math --latest-only

# 4) Upload only LLaDA 8B related folders, private repos
# python -m rl.eval.push_to_hub --model-filter llada-8b --private

# 5) Upload from a custom root path
# python -m rl.eval.push_to_hub --base-path /home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/checkpoints
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


@dataclass(frozen=True)
class UploadTarget:
    method: str
    model: str
    tag: str
    folder: Path


def _norm_name(name: str) -> str:
    """Normalize names for HF repo ids."""
    x = name.strip().lower()
    x = x.replace("/", "-")
    x = re.sub(r"[^a-z0-9._-]+", "-", x)
    x = re.sub(r"-+", "-", x).strip("-._")
    return x or "unknown"


def _contains_model_files(path: Path) -> bool:
    """Detect whether a folder looks like a complete model dir."""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists() or (
        path / "adapter_config.json"
    ).exists()
    has_weights = any(
        any(path.glob(pattern))
        for pattern in (
            "*.safetensors",
            "pytorch_model*.bin",
            "adapter_model*.safetensors",
            "model*.safetensors",
        )
    )
    return has_config and has_weights


def _iter_checkpoint_dirs(root: Path) -> Iterable[Path]:
    for p in root.rglob("checkpoint-*"):
        if p.is_dir() and p.name != "checkpoints":
            yield p


def discover_targets(base_path: Path) -> list[UploadTarget]:
    """
    Discover targets from current repo layout.

    Supported layouts:
    1) checkpoints/<method>/<model>/checkpoint-XXXX
    2) checkpoints/<method>/checkpoint-XXXX
    3) checkpoints/<method>/<final_model_dir_with_config+weights>
    """
    targets: list[UploadTarget] = []

    # 1/2) Checkpoint-style outputs.
    for ckpt_dir in _iter_checkpoint_dirs(base_path):
        rel = ckpt_dir.relative_to(base_path)
        parts = rel.parts
        if len(parts) < 2:
            continue
        method = parts[0]
        if len(parts) >= 3:
            model = parts[1]
        else:
            model = "base"
        tag = parts[-1]  # checkpoint-XXXX
        targets.append(
            UploadTarget(method=method, model=model, tag=tag, folder=ckpt_dir)
        )

    # 3) Final model directories that are not checkpoint-* children.
    for method_dir in base_path.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for child in method_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith("checkpoint-"):
                continue
            if child.name in {"runs", "logs"}:
                continue
            if _contains_model_files(child):
                targets.append(
                    UploadTarget(
                        method=method, model=child.name, tag="final", folder=child
                    )
                )

    # Deduplicate by absolute path.
    uniq = {t.folder.resolve(): t for t in targets}
    return sorted(uniq.values(), key=lambda x: str(x.folder))


def build_repo_id(namespace: str, target: UploadTarget) -> str:
    method = _norm_name(target.method)
    model = _norm_name(target.model)
    tag = _norm_name(target.tag)
    return f"{namespace}/{method}--{model}--{tag}"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_base = here.parent.parent.parent / "checkpoints"

    parser = argparse.ArgumentParser(
        description="Upload discovered checkpoints/models to HF Hub"
    )
    parser.add_argument("--base-path", type=Path, default=default_base)
    parser.add_argument("--namespace", type=str, default="diffusion-reasoning")
    parser.add_argument(
        "--methods", type=str, default="", help="Comma-separated method folder names"
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default="",
        help="Case-insensitive substring filter on model name",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Keep latest checkpoint per (method, model)",
    )
    parser.add_argument("--private", action="store_true", help="Create private repos")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print planned uploads"
    )
    return parser.parse_args()


def maybe_filter_targets(
    targets: list[UploadTarget], args: argparse.Namespace
) -> list[UploadTarget]:
    out = targets
    if args.methods:
        allow = {_norm_name(x) for x in args.methods.split(",") if x.strip()}
        out = [t for t in out if _norm_name(t.method) in allow]

    if args.model_filter:
        needle = args.model_filter.lower()
        out = [t for t in out if needle in t.model.lower()]

    if args.latest_only:
        grouped: dict[tuple[str, str], UploadTarget] = {}
        for t in out:
            key = (t.method, t.model)
            prev = grouped.get(key)
            if prev is None:
                grouped[key] = t
                continue

            def _step_num(tag: str) -> int:
                m = re.search(r"checkpoint-(\d+)$", tag)
                return int(m.group(1)) if m else -1

            if _step_num(t.tag) >= _step_num(prev.tag):
                grouped[key] = t
        out = sorted(grouped.values(), key=lambda x: (x.method, x.model, x.tag))

    return out


def main() -> None:
    args = parse_args()

    base_path = args.base_path.resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    all_targets = discover_targets(base_path)
    targets = maybe_filter_targets(all_targets, args)

    if not targets:
        print("No upload targets found after filtering")
        return

    print(f"Discovered {len(all_targets)} targets; uploading {len(targets)} targets")
    for t in targets:
        print(f"  - {t.method}/{t.model}/{t.tag} -> {t.folder}")

    if args.dry_run:
        print("Dry-run mode: no upload performed")
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Please export HF_TOKEN before uploading"
        )

    api = HfApi(token=token)
    ok = 0
    failed = 0

    for target in targets:
        repo_id = build_repo_id(args.namespace, target)
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=args.private,
            )
            print(f"[repo] ready: {repo_id}")

            api.upload_folder(
                folder_path=str(target.folder),
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[ok] uploaded: {target.folder} -> {repo_id}")
            ok += 1
        except HfHubHTTPError as e:
            print(f"[error] hub error for {repo_id}: {e}")
            failed += 1
        except Exception as e:  # keep batch upload running
            print(f"[error] failed for {repo_id}: {e}")
            failed += 1

    print(f"Done. success={ok}, failed={failed}")


if __name__ == "__main__":
    main()
