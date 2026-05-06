#!/bin/bash
ulimit -c 0

SCRIPTS=(
    # ============================================
    # d1 scripts (Diffusion-GRPO baseline)
    # ============================================
    "reproduce/grpo/grpo_countdown.sh"
    "reproduce/grpo/grpo_sudoku.sh"
    "reproduce/grpo/grpo_math.sh"
    "reproduce/grpo/grpo_gsm8k.sh"
    "reproduce/grpo/grpo_kodcode.sh"
    "reproduce/grpo/grpo_knights_and_knaves.sh"
    "reproduce/grpo/grpo_humaneval.sh"
    "reproduce/grpo/grpo_mbpp.sh"

    # ============================================
    # b1_d1 scripts (Block-based d1)
    # ============================================
    "reproduce/d1/b1_d1_countdown.sh"
    "reproduce/d1/b1_d1_sudoku.sh"
    "reproduce/d1/b1_d1_math.sh"
    "reproduce/d1/b1_d1_gsm8k.sh"
    "reproduce/d1/b1_d1_kodcode.sh"
    "reproduce/d1/b1_d1_knights_and_knaves.sh"
    "reproduce/d1/b1_d1_humaneval.sh"
    "reproduce/d1/b1_d1_mbpp.sh"

    # ============================================
    # d1_SFT scripts (d1 with SFT initialization)
    # ============================================
    "reproduce/d1/d1_SFT_countdown.sh"
    "reproduce/d1/d1_SFT_sudoku.sh"
    "reproduce/d1/d1_SFT_math.sh"
    "reproduce/d1/d1_SFT_gsm8k.sh"
    "reproduce/d1/d1_SFT_kodcode.sh"
    "reproduce/d1/d1_SFT_knights_and_knaves.sh"
    "reproduce/d1/d1_SFT_humaneval.sh"
    "reproduce/d1/d1_SFT_mbpp.sh"

    # ============================================
    # wd1 scripts (WD1 method)
    # ============================================
    "reproduce/wd1/wd1_countdown.sh"
    "reproduce/wd1/wd1_sudoku.sh"
    "reproduce/wd1/wd1_math.sh"
    "reproduce/wd1/wd1_gsm8k.sh"
    "reproduce/wd1/wd1_kodcode.sh"
    "reproduce/wd1/wd1_knights_and_knaves.sh"
    "reproduce/wd1/wd1_humaneval.sh"
    "reproduce/wd1/wd1_mbpp.sh"

    # ============================================
    # b1_wd1 scripts (Block-based WD1)
    # ============================================
    "reproduce/wd1/b1_wd1_countdown.sh"
    "reproduce/wd1/b1_wd1_sudoku.sh"
    "reproduce/wd1/b1_wd1_math.sh"
    "reproduce/wd1/b1_wd1_gsm8k.sh"
    "reproduce/wd1/b1_wd1_kodcode.sh"
    "reproduce/wd1/b1_wd1_knights_and_knaves.sh"
    "reproduce/wd1/b1_wd1_humaneval.sh"
    "reproduce/wd1/b1_wd1_mbpp.sh"

    # ============================================
    # gdpo scripts (GDPO baseline)
    # ============================================
    "reproduce/gdpo/gdpo_countdown.sh"
    "reproduce/gdpo/gdpo_sudoku.sh"
    "reproduce/gdpo/gdpo_math.sh"
    "reproduce/gdpo/gdpo_gsm8k.sh"
    "reproduce/gdpo/gdpo_kodcode.sh"
    "reproduce/gdpo/gdpo_knights_and_knaves.sh"
    "reproduce/gdpo/gdpo_humaneval.sh"
    "reproduce/gdpo/gdpo_mbpp.sh"

    # ============================================
    # b1_gdpo scripts (Block-based GDPO)
    # ============================================
    "reproduce/gdpo/b1_gdpo_countdown.sh"
    "reproduce/gdpo/b1_gdpo_sudoku.sh"
    "reproduce/gdpo/b1_gdpo_math.sh"
    "reproduce/gdpo/b1_gdpo_gsm8k.sh"
    "reproduce/gdpo/b1_gdpo_kodcode.sh"
    "reproduce/gdpo/b1_gdpo_knights_and_knaves.sh"
    "reproduce/gdpo/b1_gdpo_humaneval.sh"
    "reproduce/gdpo/b1_gdpo_mbpp.sh"

    # ============================================
    # mdpo scripts (MDPO baseline)
    # ============================================
    "reproduce/mdpo/mdpo_countdown.sh"
    "reproduce/mdpo/mdpo_sudoku.sh"
    "reproduce/mdpo/mdpo_math.sh"
    "reproduce/mdpo/mdpo_gsm8k.sh"
    "reproduce/mdpo/mdpo_kodcode.sh"
    "reproduce/mdpo/mdpo_knights_and_knaves.sh"
    "reproduce/mdpo/mdpo_humaneval.sh"
    "reproduce/mdpo/mdpo_mbpp.sh"

    # ============================================
    # b1_mdpo scripts (Block-based MDPO)
    # ============================================
    "reproduce/mdpo/b1_mdpo_countdown.sh"
    "reproduce/mdpo/b1_mdpo_sudoku.sh"
    "reproduce/mdpo/b1_mdpo_math.sh"
    "reproduce/mdpo/b1_mdpo_gsm8k.sh"
    "reproduce/mdpo/b1_mdpo_kodcode.sh"
    "reproduce/mdpo/b1_mdpo_knights_and_knaves.sh"
    "reproduce/mdpo/b1_mdpo_humaneval.sh"
    "reproduce/mdpo/b1_mdpo_mbpp.sh"

    # ============================================
    # stable_drl scripts (StableDRL method)
    # ============================================
    "reproduce/stable_drl/stable_drl_countdown.sh"
    "reproduce/stable_drl/stable_drl_sudoku.sh"
    "reproduce/stable_drl/stable_drl_math.sh"
    "reproduce/stable_drl/stable_drl_gsm8k.sh"
    "reproduce/stable_drl/stable_drl_kodcode.sh"
    "reproduce/stable_drl/stable_drl_knights_and_knaves.sh"
    "reproduce/stable_drl/stable_drl_humaneval.sh"
    "reproduce/stable_drl/stable_drl_mbpp.sh"

    # ============================================
    # espo scripts (ESPO method)
    # ============================================
    "reproduce/espo/espo_countdown.sh"
    "reproduce/espo/espo_sudoku.sh"
    "reproduce/espo/espo_math.sh"
    "reproduce/espo/espo_gsm8k.sh"
    "reproduce/espo/espo_kodcode.sh"
    "reproduce/espo/espo_knights_and_knaves.sh"
    "reproduce/espo/espo_humaneval.sh"
    "reproduce/espo/espo_mbpp.sh"
)

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