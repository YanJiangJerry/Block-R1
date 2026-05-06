import json
import os
import warnings
from rl.eval.gsm8k import GSM8KDataset

# from rl.data_utils import SYSTEM_PROMPT, SUDOKU_SYSTEM_PROMPT

# Baseline
CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)


CTD_SYSTEM_PROMPT = """You are a mathematical expert solving Countdown questions.
Your Task: Given a list of numbers and a target integer, construct an arithmetic expression that evaluates exactly to the target.
 
STRICT CONSTRAINTS ON NUMBER USAGE:
1. **NO EXTERNAL NUMBERS**: You must use ONLY and ALL the numbers provided in the input list. Do not introduce any other integers (e.g., do not use 1 or 2 unless they are explicitly in the provided list).
2. **EXACTLY ONCE**: You must use EACH number from the provided list EXACTLY ONCE. You cannot skip any number, and you cannot reuse any number multiple times.
3. **NO SPACES OR SEPARATORS**: You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
 
You should provide clear, logical reasoning step by step. Append the tag \\block directly to the end of the last sentence of each reasoning step without starting a new line.
After reasoning, provide only your final expression inside \\boxed{} tags. Do not include " = target" inside the box. Respond exactly in the following format:
<reasoning>
Step 1, ... \\block
...
Step n, ... \\block
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
    ):
        if num_examples > 0:
            warnings.warn(
                "num_examples must be 0 for Countdown dataset. Overriding num_examples to 0."
            )
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
        )  # num_examples = always 0

    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/../../dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        prompt = self.create_prompt(question)

        return prompt, question, (numbers, target)

    # def __getitem__(self, idx):
    #     # Retrieve the item based on subsample index
    #     item = self.dataset[self.subsample[idx].item()]
    #     target = int(item["output"])
    #     numbers_str = item["input"]

    #     # Parse numbers into a list of integers
    #     if isinstance(numbers_str, list):
    #         numbers = numbers_str
    #     else:
    #         # Handle cases where input is a string like "1, 2, 3"
    #         numbers = [int(num) for num in numbers_str.split(",")]

    #     # Update the question string to match the specific instruction format
    #     # defined in 'get_countdown_questions'.
    #     # We do not explicitly include SYSTEM_PROMPT here because
    #     # self.create_prompt() (from the parent class) usually prepends self.system_prompt.
    #     question = (
    #         f"Using only the numbers {numbers}, create an arithmetic expression that evaluates to exactly {target}. "
    #         f"You must use all numbers from the list, and each number must be used exactly once. "
    #         f"You may use the operations +, -, *, and / as needed. "
    #         f"After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. "
    #         f"For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"
    #     )

    #     # Create the full prompt (System Prompt + Question)
    #     prompt = self.create_prompt(question)

    #     return prompt, question, (numbers, target)