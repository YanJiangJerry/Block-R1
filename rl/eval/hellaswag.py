import numpy as np
import torch
from datasets import load_dataset
from rl.eval.mmlu import MMLUDataset, MC_SYSTEM_PROMPT

HELLASWAG_SYSTEM_PROMPT = """You are an expert at commonsense reasoning. You will be given a context and several possible endings.
Analyze each ending carefully and select the most plausible continuation.
Respond in the following format:
<reasoning>
Your analysis here
</reasoning>
<answer>
The answer is [X]
</answer>

Where [X] is the letter of the correct option (A, B, C, D)."""


def format_hellaswag_choices(endings):
    """Format endings with letter labels A-D."""
    labels = "ABCD"
    return "\n".join([f"{labels[i]}. {ending}" for i, ending in enumerate(endings)])


class HellaSwagDataset(MMLUDataset):
    """HellaSwag dataset for commonsense reasoning."""

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=HELLASWAG_SYSTEM_PROMPT,
        subsample=-1,
        split="validation",  # HellaSwag uses validation for testing
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
        print(f"Evaluating {len(self.subsample)} HellaSwag examples")
        assert subsample <= len(
            self.dataset
        ), "Subsample size is greater than dataset size"

    def load_test_dataset(self):
        """Load HellaSwag dataset from Rowan/hellaswag."""
        self.dataset = load_dataset("Rowan/hellaswag", split=self.split)
        print(f"Loaded HellaSwag dataset with {len(self.dataset)} examples")

    def get_answer_letter(self, label):
        """Convert label to letter (label is string '0'-'3')."""
        labels = "ABCD"
        return labels[int(label)]

    def create_prompt(self, ctx, activity_label, endings):
        """Create prompt for HellaSwag question."""
        formatted_choices = format_hellaswag_choices(endings)

        prompt_text = f"Activity: {activity_label}\n\nContext: {ctx}\n\nWhich is the most plausible continuation?\n\n{formatted_choices}\n\nPlease select the correct answer."

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
        ctx = item["ctx"]
        activity_label = item["activity_label"]
        endings = item["endings"]
        label = item["label"]  # String '0'-'3'
        answer = self.get_answer_letter(label)

        prompt = self.create_prompt(ctx, activity_label, endings)
        return prompt, ctx, answer, endings

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        contexts = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        endings = [item[3] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {
            "input_ids": input_ids,
            "questions": contexts,  # Use contexts as questions for consistency
            "answers": answers,
            "prompts": prompts,
            "choices": endings,
        }
