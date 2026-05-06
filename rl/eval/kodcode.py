import re
from rl.eval.gsm8k import GSM8KDataset
from datasets import Dataset as HFDataset, load_dataset


KODCODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Your task is to solve programming problems by following these steps:\n"
    "1. Analyze the problem requirements and constraints thoroughly.\n"
    "2. Think step-by-step to design an efficient algorithm.\n"
    "3. Implement the solution in clean, idiomatic Python code.\n"
    "4. Ensure the implementation strictly adheres to the provided function signature and passes all unit tests.\n"
    "All explanations must be logical and concise."
)


class KodCodeDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=KODCODE_SYSTEM_PROMPT,
        subsample=256,
    ):
        super().__init__(
            tokenizer, num_examples, add_reasoning, system_prompt, subsample
        )

    def load_test_dataset(self):
        """Load the KodCode dataset (last 500 samples for testing)."""
        dataset = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

        def extract_test_cases(test_str):
            pattern = r"^\s*(assert\s+.*)"
            asserts = re.findall(pattern, test_str, re.MULTILINE)
            return asserts

        dataset = dataset.map(lambda x: {"test_list": extract_test_cases(x["test"])})
        dataset = dataset.filter(lambda x: len(x["test_list"]) >= 3)

        # Take last 500 samples for testing (no overlap with training)
        total_len = len(dataset)
        test_size = 500
        dataset = dataset.select(range(total_len - test_size, total_len))

        self.dataset = dataset
        print(
            "Loaded Testing KodCode dataset with {} examples".format(len(self.dataset))
        )

    def create_prompt(self, input_text, test_list):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text

        str_pass_test = "Your code should pass these tests:\n\n" + "\n".join(
            test_list[:3]
        )
        content = prompt + "\n\n" + str_pass_test
        if getattr(self, "system_prompt", None):
            content = str(self.system_prompt) + "\n\n" + content
        messages = [{"role": "user", "content": content}]
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
        text = item["question"]
        test_list = item["test_list"]
        code = item["solution"]

        # Get function signature from test_info
        func_name_pars = (
            item["test_info"][0]["function_name"]
            + item["test_info"][0]["parameter_list"]
        )

        question = f"You are an expert Python programmer, and here is your task {text}\n\nFunction signature: {func_name_pars}"

        prompt = self.create_prompt(question, test_list)
        return prompt, question, code, test_list

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        test_list = [item[3] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {
            "input_ids": input_ids,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "test_list": test_list,
        }
