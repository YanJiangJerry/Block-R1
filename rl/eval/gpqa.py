import random

import numpy as np
import torch
from datasets import load_dataset


# Reuse the same MC system prompt
MC_SYSTEM_PROMPT = """You are an expert at answering multiple choice questions. You will be given a question and several options.
Analyze each option carefully and select the best answer.
Respond in the following format:
<reasoning>
Your analysis here
</reasoning>
<answer>
The answer is [X]
</answer>

Where [X] is the letter of the correct option (A, B, C, D, etc.)."""


def format_choices(choices):
    """Format choices with letter labels."""
    labels = "ABCD"
    return "\n".join([f"{labels[i]}. {choice}" for i, choice in enumerate(choices)])


class GPQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MC_SYSTEM_PROMPT,
        subsample=-1,
        split="train",
        config_name="gpqa_main",
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.split = split
        self.config_name = config_name
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"Evaluating {len(self.subsample)} GPQA examples")
        assert subsample <= len(
            self.dataset
        ), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        """Load GPQA dataset from Idavidrein/gpqa."""
        self.dataset = load_dataset(
            "Idavidrein/gpqa",
            self.config_name,
            split=self.split,
            # trust_remote_code=True,
        )
        print(
            f"Loaded GPQA ({self.config_name}) dataset with {len(self.dataset)} examples"
        )

    def _shuffle_choices(self, correct, incorrect1, incorrect2, incorrect3, seed):
        """Shuffle the 4 choices and return (choices_list, correct_letter)."""
        choices = [correct, incorrect1, incorrect2, incorrect3]
        rng = random.Random(seed)
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_idx = indices.index(0)  # original index 0 was the correct answer
        return shuffled, chr(ord("A") + correct_idx)

    def create_prompt(self, question, choices, domain=None):
        """Create prompt for GPQA question."""
        formatted_choices = format_choices(choices)
        domain_text = f"Domain: {domain}\n\n" if domain else ""
        prompt_text = f"{domain_text}Question: {question}\n\n{formatted_choices}\n\nPlease select the correct answer."

        messages = [
            {"role": "user", "content": self.system_prompt + "\n\n" + prompt_text}
        ]
        user_input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        item = self.dataset[self.subsample[idx].item()]
        question = item["Question"]
        correct = item["Correct Answer"]
        inc1 = item["Incorrect Answer 1"]
        inc2 = item["Incorrect Answer 2"]
        inc3 = item["Incorrect Answer 3"]
        domain = item.get("High-level domain", "")

        # Deterministic shuffle per example
        choices, answer = self._shuffle_choices(correct, inc1, inc2, inc3, seed=idx)

        prompt = self.create_prompt(question, choices, domain)
        return prompt, question, answer, choices

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        choices = [item[3] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {
            "input_ids": input_ids,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "choices": choices,
        }
