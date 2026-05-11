<div align="center">

<img src="Logo.png" alt="Block-R1" width="900"/>

[![Paper-b1](https://img.shields.io/badge/Paper-b1-red)](https://arxiv.org/abs/2605.02263)
[![Paper-Block--R1](https://img.shields.io/badge/Paper-Block--R1-red)](#)
[![Dataset](https://img.shields.io/static/v1?label=Dataset&message=Hugging%E2%80%8BFace&color=yellow)](https://huggingface.co/datasets/dLLM-R1/Block-R1)
[![Models](https://img.shields.io/static/v1?label=Models&message=Hugging%E2%80%8BFace&color=yellow)](https://huggingface.co/dLLM-R1/Block-R1-ckpts)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/YanJiangJerry/Block-R1)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>


## Overview

**Block-R1** is a benchmark for **multi-domain reinforcement learning with block-based diffusion large language models**, designed to enhance block-based reasoning generation in dLLMs. This codebase contains block-based reasoning datasets and the dynamic block-size generation method **b1**.

Block-R1 standardises RL training recipes, Block-R1 dataset construction, and evaluation across **reasoning, code, puzzles, and knowledge** domains, where different domains may prefer different block sizes for semi-autoregressive decoding in dLLMs.

Main components:

- **Multi-domain RL**: Train and compare the latest RL for dLLM algorithms on multiple domains and metrics under one benchmark protocol.
- **Benchmark coverage**: Diverse domains covering code, maths, puzzles, general knowledge, and advanced reasoning.
- **Block-R1 dataset construction**: Build block-based training data by comparing a student and a teacher dLLM across different block sizes.
- **Dynamic block size generation**: Support **b1**, a dynamic-size reasoning block method for dLLMs.
- **RL methods for dLLMs**: Reproduce multiple RL algorithm families under a unified codebase.
- **Backbone dLLMs**: Support LLaDA, LLaDA 1.5, LLaDA2 mini, Dream, SDAR, and TraDo.
- **Cross-vendor GPUs**: Support both NVIDIA CUDA and AMD ROCm environments.


## Catalogue

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Supported dLLM Models](#supported-dllm-models)
- [Supported RL for dLLM Methods](#supported-rl-for-dllm-methods)
- [Benchmark Domains and Data](#benchmark-domains-and-data)
- [Block-R1 Dataset](#block-r1-dataset)
- [Pipeline](#pipeline)
- [SFT](#sft)
- [Performance](#performance)
- [References and Related Resources](#references-and-related-resources)


## Key Features

- **Multi-domain RL benchmark**
  - Train and compare RL algorithms on multiple domains and metrics under one benchmark protocol.
- **Block-based dataset construction**
  - Build block-based training data by comparing model A and model B across different block sizes.
- **Dynamic-size reasoning blocks**
  - Support **b1**, a dynamic block size generation method for diffusion large language models.
- **Reproducible RL recipes**
  - Reproduce d1, GRPO, WD1, GDPO, MDPO, StableDRL, and ESPO under `reproduce/`.
- **Cross-vendor GPU support**
  - Support both NVIDIA CUDA and AMD ROCm environments.


## Installation and Setup

The main experiments in Block-R1 were run on four AMD MI300X GPUs, each with 192 GB of memory. Block-R1 also supports NVIDIA GPUs.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies for NVIDIA GPUs:

```bash
pip install -r requirements_h100.txt
```

Install dependencies for AMD GPUs:

```bash
pip install -r requirements_rocm.txt
```

Install only one of the two requirement files above for your machine class. Do not install both in the same environment.

Set data and Hugging Face cache paths:

```bash
export BASE_DATA=/path/to/data
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache
```

All scripts support SLURM systems. We recommend using at least 4 GPUs:

```bash
#SBATCH --gres=gpu:4
```

or configure GPU ids directly:

```bash
GPU_IDS=(0 1 2 3)
```

The models and datasets can be downloaded via Hugging Face using the links in the code.


## Quick Start

Clone the repository:

```bash
git clone https://anonymous.4open.science/r/Block-R1-2026/
cd Block-R1
```

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# NVIDIA
pip install -r requirements_h100.txt

# or AMD ROCm
pip install -r requirements_rocm.txt
```

Set your paths:

```bash
export BASE_DATA=/path/to/data
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache
```

Build the Block-R1 dataset:

```bash
bash block_r1_dataset.sh
```

Run multi-domain RL on Block-R1:

```bash
bash run_block_r1.sh
```

Run full RL training sweeps:

```bash
bash run_benchmark.sh
```

Evaluate backbone or RL checkpoints:

```bash
bash eval_backbone.sh
```

Evaluate GURU-style checkpoints:

```bash
bash eval_guru.sh
```


## Repository Structure

```text
Block-R1/
├── Logo.png
├── block_r1_dataset.sh
├── run_block_r1.sh
├── run_benchmark.sh
├── eval_backbone.sh
├── eval_guru.sh
├── README.md
├── requirements_h100.txt
├── requirements_rocm.txt
├── data/                              # Store all data and model from Hugging Face
├── rl/                                # Main function entry
│   ├── block_r1.py
│   ├── eval/
│   └── trainers/
│       ├── block_r1_trainer.py        # Block-R1 / R1 wrappers (dynamic block scheduling)
│       ├── diffu_grpo_trainer.py      # d1 / diffusion-GRPO (token-level clipped objective)
│       ├── wd1_grpo_trainer.py        # WD1 (NSR+PSR reweighting)
│       ├── gdpo_trainer.py            # GDPO (sequence-level clipped ratio)
│       ├── mdpo_trainer.py            # MDPO
│       ├── espo_trainer.py            # ESPO (sequence-level clipped ratio, ELBO-based)
│       ├── stable_drl_trainer.py      # StableDRL (SPG/SNIS objective)
│       ├── stable_drl_svpo.py         # StableDRL core math (SPG bound + optional SNIS)
│       ├── eval_callback.py           # periodic eval + wandb logging
│       ├── diffu_grpo_config.py       # config dataclass (CLI/yaml fields)
│       ├── likelihood_estimators.py   # GDPO logp estimators
│       ├── dynamic_generate.py        # b1 dynamic generation helpers
│       ├── cross_domain_generate.py   # cross-domain generation utilities
│       └── train_utils.py
├── reproduce/
│   ├── d1/
│   ├── grpo/
│   ├── wd1/
│   ├── gdpo/
│   ├── mdpo/
│   ├── stable_drl/
│   └── espo/
├── logs/
├── sft/
├── dataset/
├── checkpoints/
└── results/
```


## Supported dLLM Models

Block-R1 supports 10 dLLM backbone models. All training and evaluation scripts accept Hugging Face model ids.

| Family | Hugging Face model id |
| --- | --- |
| GSAI-ML / LLaDA v1 | `GSAI-ML/LLaDA-8B-Base` |
| GSAI-ML / LLaDA v1 | `GSAI-ML/LLaDA-8B-Instruct` |
| GSAI-ML / LLaDA 1.5 | `GSAI-ML/LLaDA-1.5` |
| InclusionAI / LLaDA 2 Mini | `inclusionAI/LLaDA2.0-mini` |
| InclusionAI / LLaDA 2 Mini | `inclusionAI/LLaDA2.1-mini` |
| Dream-org / Dream v0 | `Dream-org/Dream-v0-Base-7B` |
| Dream-org / Dream v0 | `Dream-org/Dream-v0-Instruct-7B` |
| JetLM / SDAR | `JetLM/SDAR-8B-Chat-b32` |
| Gen-Verse / TraDo | `Gen-Verse/TraDo-8B-Instruct` |
| Gen-Verse / TraDo | `Gen-Verse/TraDo-8B-Thinking` |

For example, `eval_backbone.sh` loops over a configurable `MODEL_PATHS` array. The default model list includes LLaDA 1.5, SDAR, TraDo, Dream-7B, and LLaDA2 mini.


## Supported RL for dLLM Methods

Block-R1 supports 7 latest RL-for-dLLM methods under `reproduce/` (one folder per method):

| Directory | Paper title |
| --- | --- |
| `reproduce/d1/` | *d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning* (Diffusion-GRPO + SFT) |
| `reproduce/grpo/` | *d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning* (Diffusion-GRPO) |
| `reproduce/wd1/` | *WD1: Weighted Policy Optimization for Reasoning in Diffusion Language Models* |
| `reproduce/gdpo/` | *Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization* |
| `reproduce/mdpo/` | *MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models* |
| `reproduce/stable_drl/` | *Stabilizing Reinforcement Learning for Diffusion Language Models* |
| `reproduce/espo/` | *Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective* |

### Dynamic block size: b1

Beyond these seven RL method families, Block-R1 supports **dynamic-size reasoning blocks** from **b1**.

Scripts prefixed with `b1_`, `block_b1_`, or `r1_b1_` under each method folder implement the dynamic block-size recipe. b1 is orthogonal to the seven algorithm folders and can be composed with them as a block scheduling or reward structure.

The corresponding paper is: *Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning*.


## Benchmark Domains and Data

Block-R1 supports 15 dataset settings. GURU follows Cheng et al., *Revisiting Reinforcement Learning for LLM Reasoning from a Cross-Domain Perspective*.

| Category | Dataset | Train size | Test size |
| --- | --- | ---: | ---: |
| Code generation | MBPP | 374 | 500 |
| Code generation | HumanEval | N/A | 164 |
| Code generation | KodCode | 9,285 | 500 |
| Mathematical reasoning | GSM8K | 7,473 | 1,319 |
| Mathematical reasoning | MATH500 | 7,500 | 500 |
| Mathematical reasoning | Countdown | 240,632 | 256 |
| Logical puzzles | Knights-and-Knaves | 6,200 | 700 |
| Logical puzzles | Sudoku | 1,000,000 | 256 |
| General capabilities | HellaSwag | 39,905 | 10,003 |
| General capabilities | MMLU | N/A | 14,042 |
| General capabilities | ARC-E | 2,251 | 2,376 |
| Advanced reasoning | MMLU-Pro | N/A | 12,032 |
| Advanced reasoning | ARC-C | 1,119 | 1,172 |
| Advanced reasoning | GPQA | N/A | 448 |
| Cross-domain RL for LLMs | GURU | 91.9K | N/A |

Eval keys in code include:

```text
gsm8k, math, countdown, sudoku, mbpp, humaneval, kodcode,
knights_and_knaves, hellaswag, mmlu, arc_e, arc_c, mmlu_pro,
gpqa
```

Additionally, GURU-aware training is supported via `reproduce/*/r1_*_guru.sh` and `eval_guru.sh`.


## Block-R1 Dataset

The Block-R1 dataset is released on Hugging Face:

```text
https://huggingface.co/datasets/dLLM-R1/Block-R1
```

The main training dataset file is:

```text
train.jsonl
```

Each sample is constructed from multi-block signals and selected according to the best A minus B block. The dataset is designed for multi-domain RL training of diffusion large language models. Please download it and place it into dataset/multi/block_r1_A_gt_B_multi_train.


## Pipeline

Block-R1 follows a complete pipeline:

```text
Block-R1 Dataset Construction -> Multi-Domain RL -> Benchmark Evaluation
```

### 1. Build the Block-R1 dataset

`block_r1_dataset.sh` is a **two-stage** driver that (1) materializes multi-block reward signals on TRAIN splits, then (2) exports a `train.jsonl` for Block-R1 training.

Stage 1 runs multi-block evaluation (via `python -m rl.block_r1 eval_multi_block ...`) and writes reward shards under the script’s `OUTPUT_DIR` (default: `./dataset/multi` under this repo).

Stage 2 exports `train.jsonl` (via `python -m rl.block_r1 build_block_r1 ...`) by selecting examples where **model A** beats **model B** at the block that maximizes \((A-B)\).

In `block_r1_dataset.sh`, the key variables you will typically edit are:

```bash
MODELS              # stage-1: backbone model list to run eval_multi_block on
DATASETS            # stage-1/2: comma-separated dataset keys (e.g., gsm8k,math,...)
BLOCK_SIZES         # stage-1/2: comma-separated block sizes
OUTPUT_DIR          # stage-1/2: output root (edit for your filesystem)
MODEL_A MODEL_B     # stage-2: pair for (A-B) selection in build_block_r1
MULTI_TRAIN_SUBDIR  # stage-2: where train.jsonl will be written under OUTPUT_DIR
```

Run:

```bash
bash block_r1_dataset.sh
```

### 2. Multi-domain RL on Block-R1

`run_block_r1.sh` launches representative Block-R1 multi-domain jobs using method entrypoints under `reproduce/`.

```bash
bash run_block_r1.sh
```

You can also pass explicit script paths to override the default list:

```bash
bash run_block_r1.sh reproduce/wd1/block_r1_wd1.sh
```

### 3. Full RL training sweeps

`run_benchmark.sh` sequentially runs a large set of RL for dLLM method training scripts under `reproduce/`.

It covers d1, GRPO, WD1, GDPO, MDPO, StableDRL, ESPO, and b1.

```bash
bash run_benchmark.sh
```

You can override the default list by passing script paths:

```bash
bash run_benchmark.sh reproduce/d1/r1_d1.sh reproduce/wd1/b1_wd1_math.sh
```

### 4. Evaluation

`eval_backbone.sh` evaluates either (a) the raw backbone (`CKPT_STEP=0`) or (b) a specific RL checkpoint (`CKPT_STEP>0`) by launching a **multi-GPU** `torch.distributed.run` job that runs `rl/eval/eval.py`.

Set `CKPT_STEP=0` to report base or instruct backbone metrics.

Set a nonzero checkpoint step and matching `METHOD` and `TRAIN_DATASET` to evaluate RL checkpoints under `checkpoints/`.

```bash
bash eval_backbone.sh
```

Configure the following variables in the script header:

```bash
MODEL_PATHS
EVAL_DATASETS
GEN_LENGTHS
GPU_IDS
CKPT_STEP
METHOD
TRAIN_DATASET
```

### 5. Optional GURU evaluation

For models trained with GURU-style run names, such as `r1_wd1_guru`, use:

```bash
bash eval_guru.sh
```

Configure:

```bash
GURU_RUN_NAME
CKPT_STEPS
MODEL_PATH
```

### 6. b1: dynamic-size block training

Scripts prefixed with `b1_*` apply the **b1** dynamic-size block mechanism on top of an existing RL recipe. They live under `reproduce/<base_method>/b1_<base_method>_<dataset>.sh`, where:

- `<base_method>` selects the underlying RL algorithm (e.g. `wd1`, `stable_drl`); the corresponding `--trainer_type` (e.g. `b1_wll` for wd1, `b1_stable_drl` for stable_drl) is set inside each script.
- `<dataset>` is one of `countdown`, `gsm8k`, `math`, `sudoku`, `kodcode`, `mbpp`, `humaneval`, `knights_and_knaves`.

Run a single recipe directly:

```bash
bash reproduce/wd1/b1_wd1_countdown.sh
bash reproduce/wd1/b1_wd1_math.sh
```

Or dispatch a subset through `run_benchmark.sh`:

```bash
bash run_benchmark.sh reproduce/wd1/b1_wd1_countdown.sh reproduce/wd1/b1_wd1_gsm8k.sh
```

Inside a `b1_*` script, the variables you typically edit are:

```bash
MODEL_NAME       # backbone (e.g. GSAI-ML/LLaDA-8B-Instruct)
DATASET          # one of countdown, gsm8k, math, sudoku, kodcode, mbpp, humaneval, knights_and_knaves
NUM_ITER         # policy-gradient inner-update iterations
RUN_NAME         # auto-built as b1_<base_method>_<dataset>
```


## SFT

Supervised fine-tuning entry points are under `sft/`.

For example:

```bash
bash sft/run_sft.sh
```

Use SFT when the corresponding recipe requires supervised fine-tuning before RL, such as d1.


## Performance

Block-R1 focuses on one-shot evaluation under both backbone and RL checkpoint settings.

The benchmark is supported to report:

- Base or instruct backbone performance.
- Single-domain RL performance.
- Multi-domain RL performance.
- Block-R1 training performance.
- b1 dynamic block-size performance.

Please refer to the paper for full experimental results.


## References and Related Resources

This benchmark builds on open-sourced RL algorithms, models, and datasets. **We sincerely thank all the authors of the works listed below for their awesome work**, which makes this benchmark possible. The included methods are:

### Methods and Algorithms

- **Diffusion-GRPO / d1**: S. Zhao et al., *d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning*, NeurIPS 2025.
- **WD1**: X. Tang et al., *WD1: Weighted Policy Optimization for Reasoning in Diffusion Language Models*, ICLR 2026.
- **GDPO**: K. Rojas et al., *Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimisation*, ICLR 2026.
- **MDPO**: H. He et al., *MDPO: Overcoming the Training-Inference Divide of Masked Diffusion Language Models*, arXiv:2508.13148, 2025.
- **SPG**: C. Wang et al., *SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models*, ICLR 2026.
- **ESPO**: J. Ou et al., *Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective*, ICLR 2026.
- **StableDRL**: J. Zhong et al., *Stabilising Reinforcement Learning for Diffusion Language Models*, arXiv:2603.06743, 2026.

### Datasets and Cross-domain RL

- **GURU**: Z. Cheng et al., *Revisiting Reinforcement Learning for LLM Reasoning from a Cross-Domain Perspective*, NeurIPS 2025.

### Dynamic-size Generation

- **b1**: Y. Jiang et al., *Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning*, ICML 2026.


## Citation

If you use this benchmark, please cite b1 and Block-R1.

```bibtex
@article{jiang2026breakblock,
  title={{Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning}},
  author={Jiang, Yan and Qiu, Ruihong and Huang, Zi},
  journal={arXiv preprint arXiv:2605.02263},
  year={2026}
}
```