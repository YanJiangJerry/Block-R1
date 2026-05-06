#!/usr/bin/env python3
"""Merge a streaming JSONL (one JSON object per line) into a pretty-printed
_generations.json file matching the format used by eval.py.

Usage:
  python merge_jsonl.py --jsonl path/to/file.jsonl --out path/to/file.json \
      [--model_path MODEL] [--checkpoint_path CP] [--gen_length N] ...

This script streams the input and does not load everything into memory.
"""
# python3 rl/eval/merge_jsonl.py \
#   --jsonl test/LLaDA-8B-Instruct_kodcode_wd1_1200/instruct_kodcode_128_64_0_generations.jsonl \
#   --out  test/LLaDA-8B-Instruct_kodcode_wd1_1200/instruct_kodcode_128_64_0_generations.json \
#   --model_path GSAI-ML/LLaDA-8B-Instruct \
#   --checkpoint_path path/to/checkpoint \
#   --gen_length 128 \
#   --diffusion_steps 64 \
#   --block_length 32 \
#   --wall_time 123.45 \
#   --total_processed 500
import argparse
import json
import os
import sys


def merge(jsonl_path, out_path, meta):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write('{\n  "generations": [\n')
        first = True
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        obj = None
                    if not first:
                        out_f.write(",\n")
                    if obj is None:
                        out_f.write("    " + line)
                    else:
                        dumped = json.dumps(obj, indent=2, ensure_ascii=True)
                        indented = "\n".join("    " + l for l in dumped.splitlines())
                        out_f.write(indented)
                    first = False

        out_f.write("\n  ],\n")

        meta_dump = json.dumps(meta, indent=2, ensure_ascii=True)
        meta_body = meta_dump[1:-1].lstrip("\n").rstrip()
        if meta_body:
            out_f.write(meta_body)
            out_f.write("\n")
        out_f.write("}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--jsonl", required=True, help="Input JSONL file (one JSON object per line)"
    )
    p.add_argument(
        "--out", required=True, help="Output JSON file path (final _generations.json)"
    )
    p.add_argument("--model_path", default="", help="Optional model path metadata")
    p.add_argument(
        "--checkpoint_path", default="", help="Optional checkpoint path metadata"
    )
    p.add_argument("--gen_length", type=int, default=None)
    p.add_argument("--diffusion_steps", type=int, default=None)
    p.add_argument("--block_length", type=int, default=None)
    p.add_argument("--wall_time", type=float, default=None)
    p.add_argument("--total_processed", type=int, default=None)

    args = p.parse_args()

    meta = {
        "metrics": {
            "wall_time": args.wall_time,
            "total_processed": args.total_processed,
        },
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "gen_length": args.gen_length,
        "diffusion_steps": args.diffusion_steps,
        "block_length": args.block_length,
    }

    if not os.path.exists(args.jsonl):
        print(f"Input JSONL {args.jsonl} does not exist", file=sys.stderr)
        sys.exit(2)

    merge(args.jsonl, args.out, meta)
    print(f"Merged {args.jsonl} -> {args.out}")
