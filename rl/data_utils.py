import os
import re
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# --- Merged from former math500_utils (math / GSM8K string helpers & rewards) ---


def boxed_in_answer(prompts, completions, answer, step=None, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        reward = 0.0
        try:
            r = r.split("<answer>")[1].split("</answer>")[0]
            reward += 1.0
        except Exception:
            reward += 0.0

        reward += 1.0 if "\boxed" in r else 0.5
        rewards.append(reward)
    return rewards


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except Exception:
        return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def create_few_shot_prompt_math(dataset, num_examples=4):
    random.seed(42)
    few_shot_examples = random.sample(range(len(dataset)), num_examples)
    formatted_examples = []
    for example in few_shot_examples:
        input_text = dataset[example]["problem"]
        answer = dataset[example]["solution"]
        formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
    prompt = "You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed\{\}. \n\n"
    return prompt + "\n\n".join(formatted_examples)


def extract_answer_first_math(generated_text):
    try:
        answer_part = generated_text
        match = re.search(r"####\s*(.*?)\s*<\|EOT\|>", answer_part)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None


def decode(tokenizer, output, skip_special_tokens=False):
    return tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)


def create_prompts(input_texts, tokenizer, few_shot_prompt=""):
    prompts = []
    for input_text in input_texts:
        m = [
            {
                "role": "user",
                "content": f"{few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n",
            }
        ]
        user_input = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        prompts.append(user_input)
    return prompts


TRAINER_TYPE = "b1_wll"


def set_trainer_type(t):
    global TRAINER_TYPE, SYSTEM_PROMPT, CTD_SYSTEM_PROMPT, SUDOKU_SYSTEM_PROMPT, GSM_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
    TRAINER_TYPE = t

    # Baseline prompts do not need \\block for training
    if "b1" not in TRAINER_TYPE:

        GSM_SYSTEM_PROMPT = (
            MATH_SYSTEM_PROMPT
        ) = """You are a math expert. You will be given a question to solve. You should provide clear, logical reasoning step by step. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

        CTD_SYSTEM_PROMPT = (
"Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
+ "{}"
+ " tags without including an equals sign or the target number. For example: <answer>a + b * c</answer>"
+ """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>"""
        )

        # Align Sudoku prompt with dLLM-ESPO eval prompt (includes an in-prompt example).
        SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

**Rules:**
- Fill empty cells with digits 1-4.
- Each row must contain digits 1-4 exactly once.
- Each column must contain digits 1-4 exactly once.
- Each 2x2 box must contain digits 1-4 exactly once.

**Important:** Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

    ######################################################################
    # Dynamic generation requires end of block marker \\block for training
    else:
        #         SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. You should provide clear, logical reasoning step by step.
        # Append the tag \\block directly to the end of the last sentence of each reasoning step without starting a new line. Respond exactly in the following format:
        # <reasoning>
        # Step 1, ... \\block
        # Step 2, ... \\block
        # ...
        # Step n, ... \\block
        # </reasoning>
        # <answer>
        # ...
        # </answer>
        # """
        SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. You should provide clear, logical reasoning step by step. Append the tag \\block directly to the end of the last sentence of each reasoning step without starting a new line. Respond exactly in the following format:
<reasoning>
Step 1, ... \\block
Step 2, ... \\block
...
Step n, ... \\block
</reasoning>
<answer>
FINAL ANSWER ONLY
</answer>
"""
        ## Math500 and GSM8K share the same prompt following rl and d1 and gdpo
        MATH_SYSTEM_PROMPT = GSM_SYSTEM_PROMPT = SYSTEM_PROMPT

        ###########################################################
        ## Countdown prompt with strict constraints on number usage
        CTD_SYSTEM_PROMPT = """You are a mathematical expert solving Countdown questions.
Your Task: Given a list of numbers and a target integer, construct an arithmetic expression that evaluates exactly to the target.

STRICT CONSTRAINTS ON NUMBER USAGE:
1. **NO EXTERNAL NUMBERS**: You must use ONLY and ALL the numbers provided in the input list. Do not introduce any other integers (e.g., do not use 1 or 2 unless they are explicitly in the provided list).
2. **EXACTLY ONCE**: You must use EACH number from the provided list EXACTLY ONCE. You cannot skip any number, and you cannot reuse any number multiple times.
3. **NO SPACES OR SEPARATORS**: You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>"

You should provide clear, logical reasoning step by step. Append the tag \\block directly to the end of the last sentence of each reasoning step without starting a new line. Respond exactly in the following format:
<reasoning>
Step 1, ... \\block
...
Step n, ... \\block
</reasoning>
<answer>
...
</answer>"""

        ###############################################################################
        ## Sudoku prompt (align with dLLM-ESPO eval prompt; no \\block requirement)
        SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

**Rules:**
- Fill empty cells with digits 1-4.
- Each row must contain digits 1-4 exactly once.
- Each column must contain digits 1-4 exactly once.
- Each 2x2 box must contain digits 1-4 exactly once.

**Important:** Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Append the tag \\block directly to the end of the last sentence of each reasoning step without starting a new line. Respond in this exact format:
<reasoning>
Step 1, ... \\block
Step 2, ... \\block
...
Step n, ... \\block
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


# Initialize with default trainer type
set_trainer_type("b1_wll")


# Follow exactly rl setting for random seed
def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": f"{GSM_SYSTEM_PROMPT}\n{x['question']}"},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


# def get_countdown_questions(split="train") -> Dataset:
#     data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
#     data = data.filter(lambda x: len(x["nums"]) == 3)


#     return data.map(
#         lambda x: {
#             "prompt": [
#                 {
#                     "role": "user",
#                     "content": f"{CTD_SYSTEM_PROMPT}\n\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
#                 },
#             ],
#             "target": x["target"],
#             "numbers": x["nums"],
#         }
#     )
# def get_countdown_questions(split="train") -> Dataset:
#     data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
#     data = data.filter(lambda x: len(x["nums"]) == 3)

#     return data.map(
#         lambda x: {
#             "prompt": [
#                 {
#                     "role": "user",
#                     # Align with dLLM-ESPO: system prompt + explicit instruction (no \\boxed requirement).
#                     "content": f"{CTD_SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\\n2*4-3\\n</answer>",
#                 },
#             ],
#             "target": x["target"],
#             "numbers": x["nums"],
#         }
#     )
def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{CTD_SYSTEM_PROMPT}\nThe provided input list is {x['nums']} and the target number is {x['target']}.\nPlease solve it according to the strict constraints above.",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


## Only math500 requires boxed for final answer during training following rl
def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    ## Original baseline prompt
                    # "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                    ## <answer>\\boxed{}</answer>
                    # "content": f"{SYSTEM_PROMPT}\nWrap the final answer in a \\boxed{{}} inside <answer></answer> tags. For example, <answer>\\boxed{{42}}</answer>. \n\n{x['problem']}",
                    ## <answer></answer>
                    "content": f"{MATH_SYSTEM_PROMPT}\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore


# ============== Prompts for MBPP/HumanEval ==============

CODE_SYSTEM_PROMPT = """You are a professional Python coding assistant.
Task: Complete the following function implementation strictly and clearly without any additional comments or explanations.
Again, implement the function strictly without any additional comments, explanations, unit tests, assertions, prints, examples in the code.

Respond in the following format:
<reasoning>
Your step-by-step reasoning here
</reasoning>
<answer>
```python
# python code only, no tests, no I/O, no prints, no asserts, no examples
```
</answer>
"""


def get_code_prompt(task, func_name_pars):
    """Generate a prompt for code completion tasks."""
    return f"""{CODE_SYSTEM_PROMPT}
Function to implement: {task}
The function name and the parameters is {func_name_pars}
"""


def extract_function_name(s: str) -> str:
    """Extract function name from a test assertion string."""
    funcs = re.findall(r"([a-zA-Z_]\w*)\s*\(", s)
    builtins = {"int", "float", "str", "list", "dict", "set", "tuple", "len", "assert"}
    funcs = [f for f in funcs if f not in builtins]
    return funcs[0] if funcs else None


def get_func_name_pars(code_str, func_name):
    """Extract function signature (name and parameters) from code."""
    pattern = r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:"
    matches = re.findall(pattern, code_str)
    for name, args in matches:
        if func_name == name:
            return f"{name}({args})"
    return ""


def get_mbpp_questions() -> Dataset:
    """Load the MBPP dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    mbpp_file_path = "../dataset/mbpp/mbpp.jsonl"
    mbpp_file_path = os.path.join(cur_path, mbpp_file_path)

    df = pd.read_json(mbpp_file_path, lines=True)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: {"func_name": extract_function_name(x["test_list"][0])})
    data = data.map(
        lambda x: {"func_name_pars": get_func_name_pars(x["code"], x["func_name"])}
    )

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": get_code_prompt(x["text"], x["func_name_pars"]),
                },
            ],
            "test_list": x["test_list"],
            "solution": x["code"],
        }
    )


def get_humaneval_questions(split: str = "test") -> Dataset:
    """Load the HumanEval dataset for evaluation."""
    # Load HumanEval (only has "test" split)
    dataset = load_dataset("openai/openai_humaneval", split=split)

    def extract_humaneval_test_cases(test_str: str, entry_point):
        """Extract test cases from HumanEval test string."""
        pattern = r"^\s*(assert\s+.*)"
        cases = re.findall(pattern, test_str, re.MULTILINE)
        cases = [re.sub(r"\bcandidate\(", f"{entry_point}(", a) for a in cases]
        return cases

    dataset = dataset.map(
        lambda x: {
            "test_list": extract_humaneval_test_cases(x["test"], x["entry_point"])
        }
    )

    return dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{CODE_SYSTEM_PROMPT}\nFunction to implement:\n{x['prompt']}",
                }
            ],
            "test_list": x["test_list"],
            "solution": x["canonical_solution"],
        }
    )


def get_kodcode_light_rl_10k(split: str = "train") -> Dataset:
    """Load the KodCode-Light-RL-10K dataset for training or testing.

    Args:
        split: "train" for training (first ~9285 samples after filtering),
               "test" for testing (last 500 samples after filtering)
    """
    dataset = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

    def extract_test_cases(test_str):
        pattern = r"^\s*(assert\s+.*)"
        asserts = re.findall(pattern, test_str, re.MULTILINE)
        return asserts

    dataset = dataset.map(lambda x: {"test_list": extract_test_cases(x["test"])})
    dataset = dataset.filter(lambda x: len(x["test_list"]) >= 3)
    dataset = dataset.map(
        lambda x: {
            "func_name_pars": x["test_info"][0]["function_name"]
            + x["test_info"][0]["parameter_list"]
        }
    )

    # Split: last 500 for test, rest for train (no overlap)
    total_len = len(dataset)
    test_size = 500
    if split == "test":
        dataset = dataset.select(range(total_len - test_size, total_len))
    else:  # train
        dataset = dataset.select(range(0, total_len - test_size))

    return dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": get_code_prompt(x["question"], x["func_name_pars"]),
                }
            ],
            "test_list": x["test_list"],
            "solution": x["solution"],
        }
    )


# ============== Prompts for Multiple Choice Tasks ==============
MC_SYSTEM_PROMPT = """You are an expert at answering multiple choice questions. Analyze each question carefully and select the best answer.
Respond in the following format:
<reasoning>
Your step-by-step reasoning here
</reasoning>
<answer>
X
</answer>
Where X is the letter of your chosen answer (A, B, C, D, etc.)."""


def format_mc_choices(choices, labels=None):
    """Format multiple choice options as A, B, C, D..."""
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
    return "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])


def get_mmlu_questions(split: str = "test") -> Dataset:
    """Load the MMLU dataset for training or evaluation."""
    # Load all subjects, ensure trust_remote_code=True
    dataset = load_dataset("cais/mmlu", "all", split=split)

    def format_question(example):
        choices_text = format_mc_choices(example["choices"])
        content = f"{MC_SYSTEM_PROMPT}\n\nQuestion: {example['question']}\n\nChoices:\n{choices_text}"
        # Convert answer index to letter
        answer_letter = chr(ord("A") + example["answer_idx"])
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": answer_letter,
            "choices": example["choices"],
        }

    # Rename 'answer' to 'answer_idx' first to allow string answer
    dataset = dataset.rename_column("answer", "answer_idx")
    return dataset.map(format_question)


def get_mmlu_pro_questions(split: str = "test") -> Dataset:
    """Load the MMLU-Pro dataset for training or evaluation."""
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    def format_question(example):
        # MMLU-Pro has 10 options (A-J)
        labels = [chr(ord("A") + i) for i in range(len(example["options"]))]
        choices_text = format_mc_choices(example["options"], labels)
        content = f"{MC_SYSTEM_PROMPT}\n\nQuestion: {example['question']}\n\nChoices:\n{choices_text}"
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": example["answer"],  # Already a letter
            "choices": example["options"],
        }

    return dataset.map(format_question)


def get_hellaswag_questions(split: str = "test") -> Dataset:
    """Load the HellaSwag dataset for training or evaluation."""
    dataset = load_dataset("Rowan/hellaswag", split=split)

    def format_question(example):
        # Context is ctx_a + ctx_b or just ctx
        context = example["ctx"]
        choices_text = format_mc_choices(example["endings"])
        content = f"{MC_SYSTEM_PROMPT}\n\nContext: {context}\n\nWhich ending best completes the context?\n\nChoices:\n{choices_text}"
        # Convert label (0-3) to letter
        answer_letter = (
            chr(ord("A") + int(example["label"])) if example["label"] != "" else "A"
        )
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": answer_letter,
            "choices": example["endings"],
        }

    return dataset.map(format_question)


def get_arc_e_questions(split: str = "test") -> Dataset:
    """Load the ARC-Easy dataset for training or evaluation."""
    dataset = load_dataset(
        "allenai/ai2_arc", "ARC-Easy", split=split
    )

    def format_question(example):
        choices_text = format_mc_choices(
            example["choices"]["text"], example["choices"]["label"]
        )
        content = f"{MC_SYSTEM_PROMPT}\n\nQuestion: {example['question']}\n\nChoices:\n{choices_text}"
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": example["answerKey"],  # Already a letter (A, B, C, D)
            "choices": example["choices"]["text"],
        }

    return dataset.map(format_question)


def get_arc_c_questions(split: str = "test") -> Dataset:
    """Load the ARC-Challenge dataset for training or evaluation."""
    dataset = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split=split
    )

    def format_question(example):
        choices_text = format_mc_choices(
            example["choices"]["text"], example["choices"]["label"]
        )
        content = f"{MC_SYSTEM_PROMPT}\n\nQuestion: {example['question']}\n\nChoices:\n{choices_text}"
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": example["answerKey"],  # Already a letter (A, B, C, D)
            "choices": example["choices"]["text"],
        }

    return dataset.map(format_question)


def get_gpqa_questions(split: str = "train", config_name: str = "gpqa_main") -> Dataset:
    """Load the GPQA dataset for evaluation.

    GPQA only has a 'train' split with 448 examples (gpqa_main).
    Answers need to be shuffled since the correct answer is always first in raw data.
    """
    import random

    dataset = load_dataset(
        "Idavidrein/gpqa", config_name, split=split
    )

    def format_question(example, idx):
        correct = example["Correct Answer"]
        choices = [
            correct,
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        # Deterministic shuffle per example
        rng = random.Random(idx)
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        answer_letter = chr(ord("A") + indices.index(0))

        choices_text = format_mc_choices(shuffled)
        content = f"{MC_SYSTEM_PROMPT}\n\nQuestion: {example['Question']}\n\nChoices:\n{choices_text}"
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": answer_letter,
            "choices": shuffled,
        }

    return dataset.map(format_question, with_indices=True)


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


def _hf_datasets_cache_dir() -> str:
    """Directory for processed dataset caches (respects HF_DATASETS_CACHE / HF_HOME)."""
    base = os.environ.get("HF_DATASETS_CACHE")
    if base:
        return base
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "datasets")


def get_knights_and_knaves_questions(split: str = "train") -> Dataset:
    """Load Knights-and-Knaves dataset from K-and-K/knights-and-knaves.
    split: 'train' or 'test'. For train, concatenates all ppl splits (2ppl-8ppl).

    Multi-GPU: use a single deterministic map cache path so all ranks share one Arrow
    cache (file-locked) instead of each process writing a full copy — avoids
    OSError Errno 122 (disk quota exceeded) on small home quotas.

    Optional: ``KNIGHTS_KNAVES_MAP_IN_MEMORY=1`` to run ``map(..., keep_in_memory=True)``
    and skip writing the post-process cache (uses more RAM).
    """
    from datasets import concatenate_datasets

    ppl_splits = ["2ppl", "3ppl", "4ppl", "5ppl", "6ppl", "7ppl", "8ppl"]
    parts = [
        load_dataset(
            "K-and-K/knights-and-knaves",
            name=split,
            split=sp,
        )
        for sp in ppl_splits
    ]
    dataset = concatenate_datasets(parts)

    def format_question(example):
        content = f"{KNIGHTS_KNAVES_SYSTEM_PROMPT}\n\n{example['quiz']}"
        return {
            # Keep prompt as plain string to avoid HF `datasets` feature incompatibilities
            # across versions (some versions serialize this as Feature type "List").
            "prompt": content,
            "answer": example["solution_text_format"],
        }

    # Single shared cache file for the formatted dataset (multi-rank safe via HF locks).
    cache_dir = os.path.join(_hf_datasets_cache_dir(), "dLLM-R1_map_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Version the cache filename so older incompatible cached schemas don't break newer runs.
    map_cache = os.path.join(cache_dir, f"knights_and_knaves_{split}_prompt_answer.v2.arrow")

    keep_mem = os.environ.get("KNIGHTS_KNAVES_MAP_IN_MEMORY", "").strip() in ("1", "true", "yes")

    # Remove original columns (solution, quiz, etc.) to avoid schema conflicts
    # when concatenating with other domains (math, kodcode) in multi-domain training
    return dataset.map(
        format_question,
        remove_columns=dataset.column_names,
        cache_file_name=map_cache,
        load_from_cache_file=True,
        num_proc=1,
        keep_in_memory=keep_mem,
    )
