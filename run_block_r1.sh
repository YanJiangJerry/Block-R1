#!/bin/bash
ulimit -c 0

SCRIPTS=(
    "reproduce/stable_drl/block_r1_stable_drl.sh"
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

# If input arguments are provided, 
# Override the default scripts list
if [ "$#" -gt 0 ]; then
    SCRIPTS=("$@")
fi

FAILED=()
for s in "${SCRIPTS[@]}"; do
    echo
    echo "----------------------------------------------------------------"
    echo "Running: $s"
    if [ ! -f "$s" ]; then
    echo "ERROR: script not found: $s"
    FAILED+=("$s (not found)")
    continue
    fi

    chmod +x "$s" 2>/dev/null || true

    if bash "$s"; then
    echo "Finished: $s"
    else
    rc=$?
    echo "Script $s exited with code $rc. Continuing to next script."
    FAILED+=("$s (exit $rc)")
    fi
done

echo
if [ ${#FAILED[@]} -ne 0 ]; then
    echo "The following scripts failed or were missing:"
    for f in "${FAILED[@]}"; do
    echo " - $f"
    done
    echo "Completed with errors. See above list."
    exit 1
else
    echo "All scripts finished successfully."
fi