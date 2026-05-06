#!/bin/bash
# =============================================================================
# eval_guru.sh — Evaluate models trained on Guru (R1), on standard benchmarks.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=/home/<YOUR_USERNAME>/<YOUR_WORKDIR>/dLLM-R1:${PYTHONPATH:-}
cd "$SCRIPT_DIR" || exit 1

module --ignore_cache load gcc/12.3.0 2>/dev/null || true

export BASE_DATA="${BASE_DATA:-data}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BASE_DATA/cache_hugg}"
export HF_HOME="${HF_HOME:-$BASE_DATA/cache_hugg}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$BASE_DATA/cache_hugg}"

# --- Manual configuration (edit here) ---
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
# Override before running: CKPT_STEPS=(0 1000 5000) bash eval_guru.sh
if [ ${#CKPT_STEPS[@]} -eq 0 ]; then
  CKPT_STEPS=(5000)
fi
GURU_RUN_NAME="${GURU_RUN_NAME:-r1_wd1_guru}"

GEN_LENGTHS=(256)
block_size=32
batch_size=64
GPU_IDS=(0 1 2 3)

# Benchmarks (all exclude guru dataset)
EVAL_DATASETS=(
  "gpqa" "hellaswag" "arc_c" "arc_e" "mmlu_pro" "mmlu"
  "gsm8k" "math" "countdown" "sudoku" "humaneval" "kodcode" "mbpp" "knights_and_knaves"
)

NUM_GPUS=${#GPU_IDS[@]}
GPU_LIST=$(IFS=, ; echo "${GPU_IDS[*]}")
MASTER_PORT=$(shuf -i 20000-30000 -n 1)
MODEL_SHORT=$(basename "$MODEL_PATH")
SAFE_MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
CKPT_BASE="checkpoints/${GURU_RUN_NAME}/${SAFE_MODEL_NAME}"

mkdir -p logs

echo "============================================================"
echo " Guru eval: CKPT_BASE=$CKPT_BASE"
echo " Model: $MODEL_PATH  CKPT_STEPS=${CKPT_STEPS[*]}"
echo " Eval sets: ${EVAL_DATASETS[*]}"
echo "============================================================"

for CKPT_STEP in "${CKPT_STEPS[@]}"; do
  if [ "$CKPT_STEP" -eq 0 ]; then
    OUTPUT_DIR="test/${MODEL_SHORT}_guru_base"
    CKPT_ARG=""
    CKPT_PATH=""
  else
    CKPT_PATH="${CKPT_BASE}/checkpoint-${CKPT_STEP}"
    if [ ! -d "$CKPT_PATH" ]; then
      echo "[WARN] Missing checkpoint: $CKPT_PATH — skip step $CKPT_STEP"
      continue
    fi
    OUTPUT_DIR="test/${MODEL_SHORT}_guru_${GURU_RUN_NAME}_${CKPT_STEP}"
    CKPT_ARG="--checkpoint_path $CKPT_PATH"
  fi

  mkdir -p "$OUTPUT_DIR"

  for eval_dataset in "${EVAL_DATASETS[@]}"; do
    for gen_length in "${GEN_LENGTHS[@]}"; do
      echo ">>> Guru eval: step=${CKPT_STEP} dataset=${eval_dataset} gen_length=${gen_length}"
      R1_ARG=""
      if [ -n "$CKPT_ARG" ] && [ -f "${CKPT_PATH}/r1.json" ]; then
        R1_ARG="--r1_controller_path ${CKPT_PATH}/r1.json"
      elif [ "$CKPT_STEP" -ne 0 ]; then
        echo "[WARN] No r1.json under ${CKPT_PATH} — R1 eval falls back to fixed block_length=${block_size}"
      fi
      if CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        -m rl.eval.r1_eval \
        --dataset "$eval_dataset" \
        --batch_size "$batch_size" \
        --block_length "$block_size" \
        --gen_length "$gen_length" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        $CKPT_ARG \
        $R1_ARG; then
        echo ">>> Done: ${eval_dataset}"
      else
        echo "[ERROR] Failed: ${eval_dataset}" | tee -a logs/guru_eval_failures.log
      fi
    done
  done

  echo "Parsing accuracy for: $OUTPUT_DIR"
  python3 -m rl.eval.parse_and_get_acc --directory "$OUTPUT_DIR" || true
done
