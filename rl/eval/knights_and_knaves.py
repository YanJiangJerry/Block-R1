"""
Knights and Knaves logical reasoning dataset from K-and-K/knights-and-knaves.
Evaluation via exact string match of <answer> content against solution_text_format.
"""
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets

KNIGHTS_KNAVES_SYSTEM_PROMPT = """You are an expert at logical reasoning. You will be given a knights and knaves puzzle.
Knights always tell the truth; knaves always lie.
Solve the puzzle step by step. You MUST put your final conclusion inside <answer> tags.
Use this exact format for the answer: (1) Name1 is a knight/knave (2) Name2 is a knight/knave ...

Respond in the following format:
<reasoning>
Your step-by-step analysis here
</reasoning>
<answer>
(1) ... (2) ...
</answer>"""


def _normalize_answer(s: str) -> str:
    """Normalize for comparison: collapse whitespace, strip."""
    if not s:
        return ""
    return " ".join(s.split()).strip()


class KnightsAndKnavesDataset(torch.utils.data.Dataset):
    """
    Load K-and-K/knights-and-knaves test set.
    Ground truth: solution_text_format for exact string comparison.
    """

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=KNIGHTS_KNAVES_SYSTEM_PROMPT,
        subsample=-1,
        split="all",
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
        print(f"Evaluating {len(self.subsample)} Knights-and-Knaves examples")
        assert subsample <= len(
            self.dataset
        ), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        """Load test set; split='all' concatenates 2ppl through 8ppl."""
        if self.split == "all":
            splits = ["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"]
            parts = [
                load_dataset("K-and-K/knights-and-knaves", "test", split=s)
                for s in splits
            ]
            self.dataset = concatenate_datasets(parts)
        else:
            self.dataset = load_dataset(
                "K-and-K/knights-and-knaves", "test", split=self.split
            )
        print(f"Loaded Knights-and-Knaves with {len(self.dataset)} examples")

    def create_prompt(self, quiz: str) -> str:
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + quiz}]
        user_input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self.add_reasoning:
            return user_input + "<reasoning>"
        return user_input

    def __getitem__(self, idx):
        item = self.dataset[self.subsample[idx].item()]
        quiz = item["quiz"]
        # solution_text_format: e.g. "(1) Zoey is a knave\n(2) Oliver is a knight"
        ground_truth = item["solution_text_format"]
        prompt = self.create_prompt(quiz)
        return prompt, quiz, ground_truth

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {
            "input_ids": input_ids,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
        }
