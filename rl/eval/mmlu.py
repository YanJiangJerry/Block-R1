import numpy as np
import torch
from datasets import load_dataset

# Multiple Choice System Prompt
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
    labels = "ABCDEFGHIJ"
    return "\n".join([f"{labels[i]}. {choice}" for i, choice in enumerate(choices)])


class MMLUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MC_SYSTEM_PROMPT,
        subsample=-1,
        split="test",
        subject="all",
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.split = split
        self.subject = subject
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"Evaluating {len(self.subsample)} MMLU examples")
        assert subsample <= len(
            self.dataset
        ), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        """Load MMLU dataset from cais/mmlu."""
        if self.subject == "all":
            self.dataset = load_dataset("cais/mmlu", "all", split=self.split)
        else:
            self.dataset = load_dataset("cais/mmlu", self.subject, split=self.split)
        print(f"Loaded MMLU dataset with {len(self.dataset)} examples")

    def get_answer_letter(self, answer_idx):
        """Convert answer index to letter."""
        labels = "ABCD"
        return labels[answer_idx]

    def create_prompt(self, question, choices, subject=None):
        """Create prompt for MMLU question."""
        formatted_choices = format_choices(choices)

        subject_text = f"Subject: {subject}\n\n" if subject else ""
        prompt_text = f"{subject_text}Question: {question}\n\n{formatted_choices}\n\nPlease select the correct answer."

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
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        answer = self.get_answer_letter(answer_idx)
        subject = item.get("subject", "general")

        prompt = self.create_prompt(question, choices, subject)
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
