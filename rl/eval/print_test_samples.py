# PYTHONPATH=. python rl/eval/print_test_samples.py --dataset math

import warnings

# Ignore warnings for cleaner output
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Import dataset classes from your existing project structure
# Ensure these files (gsm8k.py, math500.py, etc.) are in the python path
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

# Mapping dataset names to their respective classes
DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


def init_seed(seed):
    """
    Initialize random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Dump dataset questions and ground truths to a log file."
    )

    # Model path is mainly needed here to load the correct Tokenizer required by the Dataset classes
    parser.add_argument(
        "--model_path",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Path to model for tokenizer loading",
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "countdown", "sudoku", "game24"],
        default="gsm8k",
        help="The dataset to inspect",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test",
        help="Directory to save the log file",
    )

    args = parser.parse_args()

    # Initialize environment
    init_seed(42)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Tokenizer from {args.model_path}...")
    # We only need the tokenizer, not the model, for dataset initialization
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # --- 2. Dataset Initialization ---
    print(f"Initializing dataset: {args.dataset}...")

    # Set subsample to -1 to ensure we get ALL testing samples, not just a subset
    dataset_class = DATASET_MAP[args.dataset]
    dataset = dataset_class(
        tokenizer,
        subsample=-1,  # -1 usually implies 'load all' in your codebase logic
        num_examples=0,  # No few-shot examples for this dump
        add_reasoning=True,  # Maintain consistency with evaluation formatting
    )

    # Create a DataLoader
    # We use batch_size=1 to make writing to the file sequential and easy to read
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Do not shuffle, we want a deterministic order
        collate_fn=dataset.collate_fn,
    )

    # --- 3. Logging Loop ---
    output_filename = f"{args.dataset}_test_samples.log"
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Starting to write samples to: {output_path}")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        # Write Header
        f.write(f"==================================================\n")
        f.write(f"Dataset Dump: {args.dataset}\n")
        f.write(f"Total Samples: {len(dataset)}\n")
        f.write(f"==================================================\n\n")

        # Iterate through the dataloader
        for batch in tqdm(dataloader, desc="Writing test samples"):
            # Extract data from batch
            # Note: Depending on your dataset class, keys might vary slightly,
            # but based on your 'evaluate' function, keys are "questions" and "answers"
            questions = batch["questions"]
            gt_answers = batch["answers"]
            prompts = batch["prompts"]

            # Since batch_size=1, lists will have 1 element
            for q, a, p in zip(questions, gt_answers, prompts):
                count += 1
                f.write(f"Sample ID: {count}\n")
                f.write(f"--------------------------------------------------\n")
                f.write(f"Prompt:\n{p}\n")
                # f.write(f"\nQuestion:\n{q}\n")
                f.write(f"\nGround Truth:\n{a}\n")
                f.write(f"==================================================\n\n")

    print(f"Successfully wrote {count} samples to {output_path}")


if __name__ == "__main__":
    main()
