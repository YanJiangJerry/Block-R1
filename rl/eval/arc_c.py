import numpy as np
import torch
from datasets import load_dataset
from rl.eval.mmlu import MMLUDataset, MC_SYSTEM_PROMPT

ARC_C_SYSTEM_PROMPT = """You are an expert at science and reasoning. You will be given a science question and several options.
Analyze each option carefully and select the best answer.
Respond in the following format:
<reasoning>
Your analysis here
</reasoning>
<answer>
The answer is [X]
</answer>

Where [X] is the letter of the correct option (A, B, C, D, etc.)."""


def format_arc_choices(choices_data):
    """Format ARC choices with their labels."""
    labels = choices_data["label"]
    texts = choices_data["text"]
    return "\n".join([f"{label}. {text}" for label, text in zip(labels, texts)])


class ARCCDataset(MMLUDataset):
    """ARC-Challenge dataset for science reasoning."""

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=ARC_C_SYSTEM_PROMPT,
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
        print(f"Evaluating {len(self.subsample)} ARC-Challenge examples")
        assert subsample <= len(
            self.dataset
        ), "Subsample size is greater than dataset size"

    def load_test_dataset(self):
        """Load ARC-Challenge dataset from allenai/ai2_arc."""
        self.dataset = load_dataset(
            "allenai/ai2_arc", "ARC-Challenge", split=self.split
        )
        print(f"Loaded ARC-Challenge dataset with {len(self.dataset)} examples")

    def get_answer_letter(self, answer_key):
        """Answer key is already a letter (A, B, C, D, etc.)."""
        return answer_key

    def create_prompt(self, question, choices_data):
        """Create prompt for ARC question."""
        formatted_choices = format_arc_choices(choices_data)

        prompt_text = f"Question: {question}\n\n{formatted_choices}\n\nPlease select the correct answer."

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
        choices_data = item["choices"]
        answer = item["answerKey"]  # Already a letter

        # Extract choice texts for collate_fn
        choice_texts = choices_data["text"]

        prompt = self.create_prompt(question, choices_data)
        return prompt, question, answer, choice_texts

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
