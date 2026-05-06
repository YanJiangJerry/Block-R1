import numpy as np
import torch
from datasets import load_dataset
from rl.eval.arc_c import ARCCDataset, ARC_C_SYSTEM_PROMPT, format_arc_choices

ARC_E_SYSTEM_PROMPT = """You are an expert at science and reasoning. You will be given a science question and several options.
Analyze each option carefully and select the best answer.
Respond in the following format:
<reasoning>
Your analysis here
</reasoning>
<answer>
The answer is [X]
</answer>

Where [X] is the letter of the correct option (A, B, C, D, etc.)."""


class ARCEDataset(ARCCDataset):
    """ARC-Easy dataset for science reasoning."""

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=ARC_E_SYSTEM_PROMPT,
        subsample=-1,
        split="test",
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.split = split
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"Evaluating {len(self.subsample)} ARC-Easy examples")
        assert subsample <= len(
            self.dataset
        ), "Subsample size is greater than dataset size"

    def load_test_dataset(self):
        """Load ARC-Easy dataset from allenai/ai2_arc."""
        self.dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=self.split)
        print(f"Loaded ARC-Easy dataset with {len(self.dataset)} examples")
