import numpy as np
import torch
from datasets import load_dataset
from rl.eval.mmlu import MMLUDataset, MC_SYSTEM_PROMPT


def format_choices_10(options):
    """Format 10 options with letter labels A-J."""
    labels = "ABCDEFGHIJ"
    return "\n".join([f"{labels[i]}. {option}" for i, option in enumerate(options)])


class MMLUProDataset(MMLUDataset):
    """MMLU-Pro dataset with 10 options per question."""

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MC_SYSTEM_PROMPT,
        subsample=-1,
        split="test",
        category="all",
    ):
        self.category = category
        super().__init__(
            tokenizer=tokenizer,
            num_examples=num_examples,
            add_reasoning=add_reasoning,
            system_prompt=system_prompt,
            subsample=subsample,
            split=split,
            subject="all",  # Will be overridden in load_test_dataset
        )

    def load_test_dataset(self):
        """Load MMLU-Pro dataset from TIGER-Lab/MMLU-Pro."""
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=self.split)

        # Filter by category if specified
        if self.category != "all":
            self.dataset = self.dataset.filter(
                lambda x: x["category"].lower() == self.category.lower()
            )

        print(f"Loaded MMLU-Pro dataset with {len(self.dataset)} examples")

    def get_answer_letter(self, answer_str):
        """Answer is already a letter (A-J)."""
        return answer_str

    def create_prompt(self, question, options, category=None):
        """Create prompt for MMLU-Pro question."""
        formatted_choices = format_choices_10(options)

        category_text = f"Category: {category}\n\n" if category else ""
        prompt_text = f"{category_text}Question: {question}\n\n{formatted_choices}\n\nPlease select the correct answer from A to J."

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
        options = item["options"]
        answer = item["answer"]  # Already a letter A-J
        category = item.get("category", "general")

        prompt = self.create_prompt(question, options, category)
        return prompt, question, answer, options

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        options = [item[3] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {
            "input_ids": input_ids,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "choices": options,
        }
