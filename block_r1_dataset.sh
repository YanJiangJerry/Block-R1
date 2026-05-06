#!/bin/bash

# =============================================================================
# block_data.sh — Stage-2 multi-block reward evaluation on TRAIN splits
# =============================================================================
# Edit the MANUAL CONFIG section: model name, dataset list, block size list.
# Output files are written to: dLLM-R1/dataset/multi/<SAFE_MODEL_NAME>/
# =============================================================================

#SBATCH --job-name=multi_block_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=166:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:mi300x:4
#SBATCH -o ./logs/%A.multi_block.output
#SBATCH -e ./logs/%A.multi_block.error

set -euo pipefail
# ========================= ENVIRONMENT =======================================
module --ignore_cache load gcc/12.3.0
# This repo’s HF/datasets snapshot lives under <repo>/data/cache_hugg (e.g. models--GSAI-ML--LLaDA-8B-Instruct).
# Other scripts may still point at different cache roots; override if your cache is elsewhere:
#   export HF_CACHE_ROOT=/path/to/cache_hugg
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HF_CACHE="${HF_CACHE_ROOT:-${_REPO_ROOT}/data/cache_hugg}"
export HF_DATASETS_CACHE="$_HF_CACHE"
export HF_HOME="$_HF_CACHE"
export HF_HUB_CACHE="$_HF_CACHE"
# Repo root first so `rl.*` (e.g. rl.reward_func) resolves to this tree; then dLLM-R1 extras.
export PYTHONPATH="${_REPO_ROOT}:/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1${PYTHONPATH:+:${PYTHONPATH}}"
export BASE_DATA="${BASE_DATA:-data}"
# Same pattern as eval_all.sh: offline Hub/datasets (snapshots under HF_* above).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========================= DISTRIBUTED SAFETY =================================
# Knights & Knaves can have long prompts / slow batches; one lagging rank can trigger
# NCCL/RCCL watchdog timeouts. These defaults aim to make multi-GPU runs more resilient.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
# NCCL_TIMEOUT is in seconds (600s default was hit in 23780959). Increase to 1h by default.
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
# Also increase PyTorch process group timeout (seconds). Used by rl/block_r1.py _init_dist().
export TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC:-3600}"
# For long prompt evaluation, cap prompt tokens and optionally increase heartbeat verbosity.
export BLOCK_R1_MAX_PROMPT_TOKENS="${BLOCK_R1_MAX_PROMPT_TOKENS:-1024}"
export BLOCK_R1_HEARTBEAT_EVERY="${BLOCK_R1_HEARTBEAT_EVERY:-1}"

# ========================= MANUAL CONFIGURATION ==============================
# Run models sequentially (outer loop). Order matters: run 8B first.
MODELS=(
  "GSAI-ML/LLaDA-8B-Base"
  "inclusionAI/LLaDA2.0-mini"
)

# Optional: explicitly pin a Hub revision/commit for MODEL_PATH.
# If empty, rl/block_r1.py will apply its own pinned defaults (aligned with rl/run_train.py).
REVISION="${REVISION:-}"

# TRAIN datasets (NOT test): exactly these six domains supported here
DATASETS="gsm8k,math,countdown,sudoku,kodcode,knights_and_knaves"

# Candidate block sizes
BLOCK_SIZES="4,8,16,32,64,128"

# Generation settings
GEN_LENGTH=256
DIFFUSION_STEPS=128

# Subsample cap per domain (actual train sizes vary; gsm8k train is 7473 — smaller sets use all rows)
SAMPLE_N=7000
SEED=42

# Default batch size per GPU (overridden per-model below)
BATCH_SIZE=32

# GPUs
GPU_IDS=(0 1 2 3)

# Output root (under this repo unless you override)
OUTPUT_DIR="${_REPO_ROOT}/dataset/multi"

# ========================= DERIVED ===========================================
NUM_GPUS=${#GPU_IDS[@]}
GPU_LIST=$(IFS=, ; echo "${GPU_IDS[*]}")
mkdir -p logs

for MODEL_PATH in "${MODELS[@]}"; do
  # Per-model batch size (per GPU)
  if [[ "$MODEL_PATH" == "inclusionAI/LLaDA2.0-mini" ]]; then
    BATCH_SIZE=8
  elif [[ "$MODEL_PATH" == "GSAI-ML/LLaDA-8B-Base" ]]; then
    BATCH_SIZE=32
  else
    BATCH_SIZE=64
  fi

  echo "============================================================"
  echo " Multi-block TRAIN evaluation"
  echo "============================================================"
  echo " Model:           $MODEL_PATH"
  echo " Revision:        ${REVISION:-<auto>}"
  echo " Datasets:        $DATASETS"
  echo " Block sizes:     $BLOCK_SIZES"
  echo " Gen length:      $GEN_LENGTH"
  echo " Diffusion steps: $DIFFUSION_STEPS"
  echo " Sample_n/domain: $SAMPLE_N (seed=$SEED)"
  echo " Batch size:      $BATCH_SIZE"
  echo " GPUs:            $GPU_LIST ($NUM_GPUS total)"
  echo " Output dir:      $OUTPUT_DIR"
  echo " HF hub cache:    $_HF_CACHE"
  echo " Mode:            resilient (each domain × block_size in a separate accelerate subprocess)"
  echo "============================================================"

  # Resilient driver: single-process Python loops slices; each slice runs `accelerate launch` so a CUDA OOM
  # in one block size does not kill the whole job (next slice still runs). Same --seed/--sample_n keeps
  # example_id/manifest stable across slices.
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  python -m rl.block_r1 eval_multi_block \
    --resilient \
    --num_processes "$NUM_GPUS" \
    --model_path "$MODEL_PATH" \
    ${REVISION:+--revision "$REVISION"} \
    --datasets "$DATASETS" \
    --block_sizes "$BLOCK_SIZES" \
    --gen_length "$GEN_LENGTH" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --sample_n "$SAMPLE_N" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"

  # # Intermediate rank part files live under <OUTPUT_DIR>/<SAFE_MODEL>/_tmp/ (merged truth: rewards_*.jsonl).
  # # Remove after a successful run. Set CLEAN_BLOCK_TMP=0 to keep _tmp for resuming incomplete slices.
  # if [[ "${CLEAN_BLOCK_TMP:-1}" != "0" ]]; then
  #   _SAFE_MODEL="$(MODEL_PATH="$MODEL_PATH" python3 -c "from rl.block_r1 import _safe_name; import os; print(_safe_name(os.environ['MODEL_PATH'].replace('/','-')))")"
  #   _TMP_DIR="${OUTPUT_DIR}/${_SAFE_MODEL}/_tmp"
  #   if [[ -d "$_TMP_DIR" ]]; then
  #     rm -rf "$_TMP_DIR"
  #     echo "[block_data] removed intermediate dir: ${_TMP_DIR}"
  #   fi
  # fi
  
done

# Start to assemble the block-r1 dataset
set -euo pipefail
source activate /home/<YOUR_USERNAME>/<YOUR_WORKDIR>/envs/reason
export PYTHONPATH=/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1:${PYTHONPATH:-}

# ========================= MANUAL CONFIGURATION ==============================
# A / B: reward comparison is (A - B); we keep examples where A wins at the argmax-(A-B) block.
MODEL_A="inclusionAI/LLaDA2.0-mini"
MODEL_B="GSAI-ML/LLaDA-8B-Base"

DATASETS="gsm8k,math,countdown,sudoku,kodcode,knights_and_knaves"
BLOCK_SIZES="4,8,16,32,64,128"

OUTPUT_DIR="/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/dataset/multi"
MULTI_TRAIN_SUBDIR="${MULTI_TRAIN_SUBDIR:-block_r1_A_gt_B_multi_train}"
RESUME="${RESUME:-1}"  # set to 1 to append/skip completed domains

mkdir -p logs

echo "============================================================"
echo " Block-R1 multi-domain train export (A>B at best A-B block)"
echo "============================================================"
echo " Model A:          $MODEL_A"
echo " Model B:          $MODEL_B"
echo " Datasets:         $DATASETS"
echo " Block sizes:      $BLOCK_SIZES"
echo " Output dir:       $OUTPUT_DIR"
echo " Train JSONL:      $OUTPUT_DIR/$MULTI_TRAIN_SUBDIR/train.jsonl"
echo " META:             $OUTPUT_DIR/$MULTI_TRAIN_SUBDIR/META.json"
echo "============================================================"

python3 -m rl.block_r1 build_block_r1 \
  --model_a "$MODEL_A" \
  --model_b "$MODEL_B" \
  --datasets "$DATASETS" \
  --block_sizes "$BLOCK_SIZES" \
  --output_dir "$OUTPUT_DIR" \
  --multi_train_subdir "$MULTI_TRAIN_SUBDIR" \
  $([[ "$RESUME" == "1" ]] && echo "--resume") \
  $([[ "${DEDUP_SUBSTRING:-0}" == "1" ]] && echo "--dedup_substring")