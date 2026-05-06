#!/bin/bash
# =============================================================================
# manual_eval.sh — Manually configure model, checkpoint, and datasets for eval
# =============================================================================
# Usage:
#   bash manual_eval.sh
#
# After completion, parse accuracy with:
#   python3 -m rl.eval.parse_and_get_acc --directory test/<OUTPUT_DIR_NAME>
#
# The output dir is automatically named as:
#   test/{MODEL_SHORT}_{TRAIN_DATASET}_{METHOD}_{CKPT_STEP}
# e.g. test/LLaDA-8B-Instruct_gsm8k_wd1_1400
#      test/LLaDA-8B-Instruct_gsm8k_wd1_base   (when CKPT_STEP=0)
# =============================================================================

#SBATCH --job-name=manual_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=166:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:mi300x:4
#SBATCH -o ./logs/%A.output
#SBATCH -e ./logs/%A.error

# ========================= ENVIRONMENT =======================================
module --ignore_cache load gcc/12.3.0
# module --ignore_cache load cuda/12.2.0
# source activate /home/<YOUR_USERNAME>/<YOUR_WORKDIR>/envs/rocm

export PYTHONPATH=/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1:${PYTHONPATH:-}
export BASE_DATA="${BASE_DATA:-data}"
# Point HF cache to the location where datasets are actually stored
export HF_DATASETS_CACHE=/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/data/cache_hugg
export HF_HOME=/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/data/cache_hugg
export HF_HUB_CACHE=/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/data/cache_hugg

# ========================= MANUAL CONFIGURATION ==============================
# 1. Model name / HuggingFace path
# MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
# MODEL_PATH="Dream-org/Dream-v0-Instruct-7B"
# Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE to 1 to run offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODEL_PATH="inclusionAI/LLaDA2.0-mini"
# MODEL_PATH="inclusionAI/LLaDA2.1-mini"
# The following models are supported:
# ==========================================
# GSAI-ML / LLaDA Family (v1 - 8B)
# ==========================================
# The classic 8B diffusion models
# MODEL_NAME=GSAI-ML/LLaDA-8B-Base
# MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct

# ==========================================
# GSAI-ML / LLaDA 1.5
# ==========================================
# Improved alignment version using VRPO
# MODEL_NAME=GSAI-ML/LLaDA-1.5

# ==========================================
# InclusionAI / LLaDA 2 Mini Series
# ==========================================
# Lightweight implementations of LLaDA 2 architecture
# Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE to 1 to run offline
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# MODEL_NAME=inclusionAI/LLaDA2.0-mini
# MODEL_NAME=inclusionAI/LLaDA2.1-mini

# ==========================================
# Dream-org / Dream Family (v0 - 7B)
# ==========================================
# Standard Text Generation
# MODEL_NAME=Dream-org/Dream-v0-Base-7B
# MODEL_NAME=Dream-org/Dream-v0-Instruct-7B

# ==========================================
# JetLM / SDAR Family (v0 - 8B)
# ==========================================
# Adapting AR model to dLLM models like Dream
# MODEL_NAME=JetLM/SDAR-8B-Chat-b32
# MODEL_NAME=Gen-Verse/TraDo-8B-Instruct
# MODEL_NAME=Gen-Verse/TraDo-8B-Thinking


# Optional: explicitly pin a Hub revision/commit for MODEL_PATH.
# If empty, rl/eval/eval.py will apply its own pinned defaults (aligned with rl/run_train.py).
REVISION="${REVISION:-}"

# 2. Checkpoint step(s) (set a list; use 0 to use the base model WITHOUT any LoRA checkpoint)
# Provide one or more steps. These will be paired 1:1 with TRAIN_DATASETS below.
# Example: CKPT_STEPS=(0) for base model
CKPT_STEPS=(0)
# llada-8b-instruct steps: 1600 (math), 900 (sudoku), 1900 (countdown), 1800 (gsm8k), 1900 (humaneval)
# CKPT_STEPS=(1600 900 1900 1800 1900 200)
# dream-v0-instruct-7b steps: 1600 (math), 1200 (sudoku), 2700 (countdown), 2700 (gsm8k), 1800 (humaneval), 3600 (kodcode)
# CKPT_STEPS=(2700 2700 1800 3600)

# 3. Training method prefix (the RUN_NAME used during training, before _${DATASET})
#    e.g. "b1_wd1", "wd1", "d1_us", "b1_d1", "gdpo", "b1_gdpo", "wll_P", "wll_SFT_NP"
METHOD="wd1"

# 4. Training dataset name(s) (used to locate checkpoint dir AND to name the output)
# Provide one or more dataset names. Must correspond 1:1 with CKPT_STEPS above.
# Example: TRAIN_DATASETS=("countdown" "gsm8k" "kodcode")
# TRAIN_DATASETS=("countdown" "gsm8k" "humaneval" "kodcode")
TRAIN_DATASETS=("countdown")

# 5. List of eval (test) datasets — must be valid keys in DATASET_MAP of eval.py
#    Supported: gsm8k math countdown sudoku mbpp humaneval kodcode mmlu mmlu_pro hellaswag arc_c arc_e gpqa knights_and_knaves
# EVAL_DATASETS=("gpqa" "hellaswag" "arc_c" "arc_e" "mmlu_pro" "mmlu" "gsm8k" "math" "countdown" "sudoku" "humaneval" "kodcode" "mbpp" "knights_and_knaves")
# EVAL_DATASETS=("gsm8k" "math" "countdown" "sudoku" "humaneval" "kodcode" "mbpp" "knights_and_knaves")
EVAL_DATASETS=("humaneval" "kodcode" "mbpp" "knights_and_knaves" "gpqa" "hellaswag" "arc_c" "arc_e" "mmlu_pro" "mmlu")

# 6. Generation / inference settings
GEN_LENGTHS=(256)
block_size=32       # block length for diffusion
batch_size=64       # reduce if OOM

# 7. GPU settings
GPU_IDS=(0 1 2 3)
# ========================= DERIVED VARIABLES =================================
NUM_GPUS=${#GPU_IDS[@]}
GPU_LIST=$(IFS=, ; echo "${GPU_IDS[*]}")
MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# Short model name (last component of path, e.g. "LLaDA-8B-Instruct")
MODEL_SHORT=$(basename "$MODEL_PATH")
# Safe model name for checkpoint directory (e.g. "GSAI-ML-LLaDA-8B-Instruct")
SAFE_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')


# Sanity: CKPT_STEPS and TRAIN_DATASETS must be 1:1
if [ ${#CKPT_STEPS[@]} -ne ${#TRAIN_DATASETS[@]} ]; then
  echo "ERROR: CKPT_STEPS (length=${#CKPT_STEPS[@]}) and TRAIN_DATASETS (length=${#TRAIN_DATASETS[@]}) must have the same length"
  exit 1
fi

mkdir -p logs

echo "============================================================"
echo " Eval Configuration"
echo "============================================================"
echo " Model:            $MODEL_PATH"
echo " Revision:         ${REVISION:-<auto>}"
echo " Method:           $METHOD"
echo " CKPT steps:       ${CKPT_STEPS[*]}"
echo " Train datasets:   ${TRAIN_DATASETS[*]}"
echo " Eval datasets:    ${EVAL_DATASETS[*]}"
echo " Gen lengths:      ${GEN_LENGTHS[*]}"
echo " Block size:       $block_size"
echo " Batch size:       $batch_size"
echo " GPUs:             $GPU_LIST ($NUM_GPUS total)"
echo " Master port:      $MASTER_PORT"
echo "============================================================"

# ========================= RUN EVALUATIONS ===================================
# Iterate over paired CKPT_STEPS and TRAIN_DATASETS (1:1 mapping)
for idx in "${!CKPT_STEPS[@]}"; do
  CKPT_STEP=${CKPT_STEPS[$idx]}
  TRAIN_DATASET=${TRAIN_DATASETS[$idx]}

  CKPT_BASE="checkpoints/${METHOD}_${TRAIN_DATASET}/${SAFE_MODEL_NAME}"

  if [ "$CKPT_STEP" -eq 0 ]; then
    OUTPUT_DIR="test/${MODEL_SHORT}_base"
    CKPT_ARG=""
    CKPT_DISPLAY="(base model, no LoRA)"
  else
    CKPT_PATH="${CKPT_BASE}/checkpoint-${CKPT_STEP}"

    # If checkpoint missing, skip this pair but continue others
    if [ ! -d "$CKPT_PATH" ]; then
      echo "[WARN] Checkpoint directory does not exist: $CKPT_PATH — skipping this ckpt/train pair"
      echo "Available checkpoints under ${CKPT_BASE}/:"
      ls -d "${CKPT_BASE}"/checkpoint-* 2>/dev/null || echo "  (none found)"
      echo "$(date) SKIPPED: ${TRAIN_DATASET} checkpoint ${CKPT_STEP}" >> logs/failures.log
      continue
    fi

    OUTPUT_DIR="test/${MODEL_SHORT}_${TRAIN_DATASET}_${METHOD}_${CKPT_STEP}"
    CKPT_ARG="--checkpoint_path $CKPT_PATH"
    CKPT_DISPLAY="($CKPT_PATH)"
  fi

  mkdir -p "$OUTPUT_DIR"

  # Run evaluations for this ckpt/train combination
  for eval_dataset in "${EVAL_DATASETS[@]}"; do
    for gen_length in "${GEN_LENGTHS[@]}"; do
      echo ""
      echo ">>> Evaluating: train=${TRAIN_DATASET} step=${CKPT_STEP} ${CKPT_DISPLAY} dataset=${eval_dataset} gen_length=${gen_length}"

      if CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        rl/eval/eval.py \
        --dataset "$eval_dataset" \
        --batch_size "$batch_size" \
        --block_length "$block_size" \
        --gen_length "$gen_length" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        ${REVISION:+--revision "$REVISION"} \
        $CKPT_ARG; then
        echo ">>> Done: train=${TRAIN_DATASET} step=${CKPT_STEP} dataset=${eval_dataset} gen_length=${gen_length}"
      else
        echo "[ERROR] Eval failed: train=${TRAIN_DATASET} step=${CKPT_STEP} dataset=${eval_dataset} gen_length=${gen_length}"
        echo "$(date) FAILED: train=${TRAIN_DATASET} step=${CKPT_STEP} dataset=${eval_dataset} gen_length=${gen_length}" >> logs/failures.log
        # continue to next gen_length/dataset
        continue
      fi

    done
  done

  # After finishing evaluations for this output dir, attempt to parse accuracy (don't let parsing failure stop other pairs)
  echo "Parsing accuracy for: $OUTPUT_DIR"
  if ! python3 -m rl.eval.parse_and_get_acc --directory "$OUTPUT_DIR"; then
    echo "[WARN] Parsing accuracy failed for $OUTPUT_DIR" >> logs/failures.log
  fi
done

## Parse accuracy for a specific output directory
# python3 -m rl.eval.parse_and_get_acc --directory /home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/test/LLaDA-8B-Instruct_kodcode_wd1_1200