#!/bin/bash

#SBATCH --job-name=stable_drl
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

# module --ignore_cache load gcc/12.3.0
# module --ignore_cache load cuda/12.2.0

# source $EBROOTANACONDA3/etc/profile.d/conda.sh
# eval "$(conda shell.bash hook)"
# module load anaconda3/2022.05

# source activate /home/<YOUR_USERNAME>/<YOUR_WORKDIR>/envs/reason

export BASE_DATA="${BASE_DATA:-data}"
echo "Saving to $BASE_DATA"

export VAR_DATA=$BASE_DATA/var_diff

export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA

export LOGDIR=$BASE_DATA/var_diff/logs
mkdir -p $LOGDIR

export WANDB_PROJECT=stable_drl

# ==========================================
# GSAI-ML / LLaDA Family (v1 - 8B)
# ==========================================
# The classic 8B diffusion models
# MODEL_NAME=GSAI-ML/LLaDA-8B-Base
MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct
# MODEL_NAME=diffusion-reasoning/LLaDA-8B-Instruct-SFT

# ==========================================
# GSAI-ML / LLaDA 1.5
# ==========================================
# Improved alignment version using VRPO
# MODEL_NAME=GSAI-ML/LLaDA-1.5

# ==========================================
# InclusionAI / LLaDA 2 Mini Series
# ==========================================
# Lightweight implementations of LLaDA 2 architecture
# Perfect for consumer GPUs and edge testing
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

DATASET="kodcode"
RUN_NAME=stable_drl_${DATASET}
NUM_ITER=12 # number of policy gradient inner updates iterations
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
    --trainer_type stable_drl \
    --run_name $RL_RUN_NAME \
    --wandb_project $WANDB_PROJECT \
    --output_dir checkpoints/${RUN_NAME}/${SAFE_MODEL_NAME} \
    2>&1 | tee -a $LOGDIR/$RUN_NAME.log
