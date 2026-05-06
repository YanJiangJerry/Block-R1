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

# 1. Model names / HuggingFace paths (list — will loop through all)
MODEL_PATHS=(
    "GSAI-ML/LLaDA-8B-Instruct"
    "GSAI-ML/LLaDA-1.5"
    "JetLM/SDAR-8B-Chat-b32"
    "Gen-Verse/TraDo-8B-Instruct"
    "Gen-Verse/TraDo-8B-Thinking"
    "Dream-org/Dream-v0-Instruct-7B"
    "inclusionAI/LLaDA2.0-mini"
    "inclusionAI/LLaDA2.1-mini"
)

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

# 2. Checkpoint step number (set to 0 to use the base model WITHOUT any LoRA checkpoint)
CKPT_STEP=0

# 3. Training method prefix (the RUN_NAME used during training, before _${DATASET}), if ckpt_step is 0, then use the base model without any LoRA checkpoint
#    e.g. "b1_wd1", "wd1", "d1_us", "b1_d1", "gdpo", "b1_gdpo", "wll_P", "wll_SFT_NP"
METHOD="wd1"

# 4. Training dataset name (used to locate checkpoint dir AND to name the output)
#    e.g. "gsm8k", "math", "countdown", "sudoku", "kodcode", "knights_and_knaves"
TRAIN_DATASET="gsm8k"

# 5. List of eval (test) datasets — must be valid keys in DATASET_MAP of eval.py
#    Supported: "countdown" "sudoku" "mbpp" "humaneval" "kodcode" "gsm8k" "math" "gpqa" "hellaswag" "arc_c" "arc_e" "mmlu_pro" "mmlu" "knights_and_knaves"
EVAL_DATASETS=("knights_and_knaves")

# 6. Generation / inference settings
GEN_LENGTHS=(256)
block_size=32       # block length for diffusion
batch_size=32       # reduce if OOM

# 7. GPU settings
GPU_IDS=(0 1 2 3)
# ========================= DERIVED VARIABLES =================================
NUM_GPUS=${#GPU_IDS[@]}
GPU_LIST=$(IFS=, ; echo "${GPU_IDS[*]}")

mkdir -p logs

# ========================= LOOP OVER MODELS ==================================
for MODEL_PATH in "${MODEL_PATHS[@]}"; do

MASTER_PORT=$(shuf -i 20000-30000 -n 1)

# Short model name (last component of path, e.g. "LLaDA-8B-Instruct")
MODEL_SHORT=$(basename "$MODEL_PATH")
# Safe model name for checkpoint directory (e.g. "GSAI-ML-LLaDA-8B-Instruct")
SAFE_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')

# Checkpoint directory: checkpoints/{METHOD}_{TRAIN_DATASET}/{SAFE_MODEL_NAME}/checkpoint-{CKPT_STEP}
CKPT_BASE="checkpoints/${METHOD}_${TRAIN_DATASET}/${SAFE_MODEL_NAME}"

# Build output directory name
if [ "$CKPT_STEP" -eq 0 ]; then
    OUTPUT_DIR="test/${MODEL_SHORT}_base"
    CKPT_ARG=""   # no --checkpoint_path → eval.py uses the raw base/instruct model
else
    OUTPUT_DIR="test/${MODEL_SHORT}_${TRAIN_DATASET}_${METHOD}_${CKPT_STEP}"
    CKPT_PATH="${CKPT_BASE}/checkpoint-${CKPT_STEP}"

    # Sanity-check: does the checkpoint directory exist?
    if [ ! -d "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint directory does not exist: $CKPT_PATH"
        echo "Available checkpoints under ${CKPT_BASE}/:"
        ls -d "${CKPT_BASE}"/checkpoint-* 2>/dev/null || echo "  (none found)"
        echo "Skipping model: $MODEL_PATH"
        continue
    fi
    CKPT_ARG="--checkpoint_path $CKPT_PATH"
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo " Eval Configuration — Model: $MODEL_PATH"
echo "============================================================"
echo " Train dataset:    $TRAIN_DATASET"
echo " Checkpoint step:  ${CKPT_STEP} $([ $CKPT_STEP -eq 0 ] && echo '(base model, no LoRA)' || echo "($CKPT_PATH)")"
echo " Eval datasets:    ${EVAL_DATASETS[*]}"
echo " Gen lengths:      ${GEN_LENGTHS[*]}"
echo " Block size:       $block_size"
echo " Batch size:       $batch_size"
echo " GPUs:             $GPU_LIST ($NUM_GPUS total)"
echo " Output dir:       $OUTPUT_DIR"
echo " Master port:      $MASTER_PORT"
echo "============================================================"

# ========================= RUN EVALUATIONS ===================================
for eval_dataset in "${EVAL_DATASETS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    echo ""
    echo ">>> [$MODEL_SHORT] Evaluating: dataset=$eval_dataset  gen_length=$gen_length  step=$CKPT_STEP"

    CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m torch.distributed.run \
      --nproc_per_node=$NUM_GPUS \
      --master_port=$MASTER_PORT \
      rl/eval/eval.py \
      --dataset "$eval_dataset" \
      --batch_size "$batch_size" \
      --block_length "$block_size" \
      --gen_length "$gen_length" \
      --output_dir "$OUTPUT_DIR" \
      --model_path "$MODEL_PATH" \
      $CKPT_ARG

    echo ">>> [$MODEL_SHORT] Done: $eval_dataset @ gen_length=$gen_length"
  done
done

# # ========================= PARSE ACCURACY ====================================
# echo ""
# echo "============================================================"
# echo " [$MODEL_SHORT] All evaluations completed! Parsing accuracy..."
# echo "============================================================"
# python3 -m rl.eval.parse_and_get_acc --directory "$OUTPUT_DIR"
# echo ""
# echo "To re-run accuracy parsing later:"
# echo "  python3 -m rl.eval.parse_and_get_acc --directory $OUTPUT_DIR"

done  # end MODEL_PATHS loop

echo ""
echo "============================================================"
echo " All models finished!"
echo "============================================================"

# Parse accuracy for a specific output directory
python3 -m rl.eval.parse_and_get_acc --directory /home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1/test/LLaDA-8B-Instruct_kodcode_wd1_1200