#!/bin/bash
# StableDRL + b1 prompts (\\block). Same algorithm as stable_drl; only prompt format switches to b1.

#SBATCH --job-name=b1_stable_drl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_sxm
#SBATCH --qos=sxm
#SBATCH --gres=gpu:h100:4
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH -o ./logs/%A.output
#SBATCH -e ./logs/%A.error

export BASE_DATA="${BASE_DATA:-data}"
export VAR_DATA=$BASE_DATA/b1_stable_drl
export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA
export LOGDIR=$BASE_DATA/b1_stable_drl/logs
mkdir -p $LOGDIR

export WANDB_PROJECT=b1_stable_drl

# MODEL_NAME=inclusionAI/LLaDA2.0-mini
MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct
DATASET="sudoku"
RUN_NAME=b1_stable_drl_${DATASET}
NUM_ITER=8
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '-')
RL_RUN_NAME=${RUN_NAME}_${SAFE_MODEL_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file "rl/accelerate.yaml" \
    --num_processes 4 \
    --main_process_port $((RANDOM % 10000 + 20000)) rl/run_train.py \
    --config "rl/train.yaml" \
    --model_path "$MODEL_NAME" \
    --num_iterations "$NUM_ITER" \
    --dataset "$DATASET" \
    --trainer_type b1_stable_drl \
    --run_name "$RL_RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    --output_dir "checkpoints/${RUN_NAME}/${SAFE_MODEL_NAME}" \
    2>&1 | tee -a "$LOGDIR/$RUN_NAME.log"

