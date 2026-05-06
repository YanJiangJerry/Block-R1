#!/bin/bash

#SBATCH --job-name=dLLM_SFT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_sxm
#SBATCH --qos=sxm
#SBATCH --gres=gpu:h100:4
#SBATCH --account=a_eecs_ds
#SBATCH -o ./logs/%A_sft.output
#SBATCH -e ./logs/%A_sft.error

module --ignore_cache load gcc/12.3.0
module --ignore_cache load cuda/12.2.0

# Environment setup
export BASE_DATA="${BASE_DATA:-data}"
echo "Saving to $BASE_DATA"

export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA/wandb

mkdir -p logs

# Model and training configuration
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
# MODEL_NAME="GSAI-ML/LLaDA-1.5"
TRAIN_DATA="simplescaling/s1K"
JOB_NAME="llada_sft_s1k"
OUTPUT_DIR="${BASE_DATA}/sft_checkpoints"

# Run with accelerate (DeepSpeed ZeRO-2)
accelerate launch \
    --config_file sft/ddp_config.yaml \
    --num_processes 4 \
    sft/sft_train.py \
    --model_name $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --job_name $JOB_NAME \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --max_length 4096 \
    --num_epochs 20 \
    --learning_rate 1e-5 \
    2>&1 | tee -a logs/${JOB_NAME}.log
