#!/bin/bash
# Block-R1 + b1 prompt format (\\block) + StableDRL.
# Each sample still uses its own br1_best_block_size from the offline Block-R1 JSONL.
# Prereq: build train.jsonl via block_r1.sh (or equivalent build_block_r1).

#SBATCH --job-name=block_b1_stable_drl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=160:00:00
#SBATCH --partition=gpu_sxm
#SBATCH --qos=sxm
#SBATCH --gres=gpu:h100:4
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH -o ./logs/%A.output
#SBATCH -e ./logs/%A.error

module --ignore_cache load gcc/12.3.0
module --ignore_cache load cuda/12.2.0

export BASE_DATA="${BASE_DATA:-data}"
export VAR_DATA=$BASE_DATA/r1_diff
export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA
export LOGDIR=$BASE_DATA/r1_diff/logs
mkdir -p "$LOGDIR"

export WANDB_PROJECT=block_b1_stable_drl
# export WANDB_MODE=offline
# export WANDB_DISABLED=false

MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"

# Default: output of block_r1.sh (MULTI_TRAIN_SUBDIR=block_r1_A_gt_B_multi_train)
BLOCK_R1_TRAIN_JSONL="${BLOCK_R1_TRAIN_JSONL:-${PWD}/dataset/multi/block_r1_A_gt_B_multi_train/train.jsonl}"

R1_DOMAINS="${R1_DOMAINS:-gsm8k,math,countdown,sudoku,kodcode,knights_and_knaves}"

RUN_NAME=block_b1_stable_drl_multi
NUM_ITER=12
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '-')
RL_RUN_NAME=${RUN_NAME}_${SAFE_MODEL_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file "rl/accelerate.yaml" \
    --num_processes 4 \
    --main_process_port $((RANDOM % 10000 + 20000)) rl/run_block_r1.py \
    --config "rl/train.yaml" \
    --model_path "$MODEL_NAME" \
    --num_iterations "$NUM_ITER" \
    --dataset gsm8k \
    --trainer_type br1_b1_stable_drl \
    --use_r1 true \
    --use_block_r1_dataset true \
    --block_r1_train_jsonl "$BLOCK_R1_TRAIN_JSONL" \
    --r1_domains "$R1_DOMAINS" \
    --run_name "$RL_RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    --log_completions true \
    --output_dir "checkpoints/${RUN_NAME}/${SAFE_MODEL_NAME}" \
    2>&1 | tee -a "$LOGDIR/$RUN_NAME.log"
