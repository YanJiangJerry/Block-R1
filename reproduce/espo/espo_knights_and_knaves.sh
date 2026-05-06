#!/bin/bash

#SBATCH --job-name=espo
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
echo "Saving to $BASE_DATA"

export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA

export LOGDIR=$BASE_DATA/espo/logs
mkdir -p $LOGDIR

export WANDB_PROJECT=espo

MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct

DATASET="knights_and_knaves"
RUN_NAME=espo_${DATASET}
NUM_ITER=12
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '-')
RL_RUN_NAME=${RUN_NAME}_${SAFE_MODEL_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file "rl/accelerate.yaml" \
    --num_processes 4 \
    --main_process_port $((RANDOM % 10000 + 20000)) rl/run_train.py \
    --config "rl/train.yaml" \
    --model_path $MODEL_NAME \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --trainer_type espo \
    --espo_num_mc 2 \
    --espo_reduce_var true \
    --run_name $RL_RUN_NAME \
    --wandb_project $WANDB_PROJECT \
    --output_dir checkpoints/${RUN_NAME}/${SAFE_MODEL_NAME} \
    2>&1 | tee -a $LOGDIR/$RUN_NAME.log

