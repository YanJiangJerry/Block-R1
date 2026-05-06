import json
import logging
import os
import re
import ast
import subprocess
import warnings
from collections.abc import Mapping
from typing import Any, List, Optional

_logger = logging.getLogger(__name__)

import numpy as np
# NOTE: Keep backward-compatible imports for scripts that run from within `rl/`
# (where `data_utils.py` is importable as a top-level module), while also
# supporting package execution like `python -m rl.block_r1` where imports must
# be qualified as `rl.*`.
try:
    from data_utils import (  # type: ignore
        boxed_in_answer,
        is_equiv,
        last_boxed_only_string,
        remove_boxed,
    )
except ModuleNotFoundError:
    from rl.data_utils import (
        boxed_in_answer,
        is_equiv,
        last_boxed_only_string,
        remove_boxed,
    )


# For gsm8k and math500
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# For gsm8k
def correctness_reward_func(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    # Only print for first example in per_device_batch_size
    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


# For countdown
def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


# For countdown
def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Extracted solution: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print("Error evaluating equation")
        return format_score


# For countdown
def _countdown_ground_truth_for_index(i: int, kwargs: dict) -> dict:
    """
    Training passes batched fields (list per column). Block-R1 passes one row:
    target may be a scalar int and numbers may be a single puzzle [n1, n2, ...].
    """
    tgt = kwargs["target"]
    nums = kwargs["numbers"]
    if isinstance(tgt, (list, tuple)):
        t_i = tgt[i]
    else:
        t_i = tgt
    if isinstance(nums, (list, tuple)) and len(nums) > 0 and isinstance(nums[0], (list, tuple)):
        n_i = list(nums[i])
    else:
        n_i = list(nums)
    return {"target": t_i, "numbers": n_i}


def countdown_reward_func(
    prompts, completions, run_name=None, step=None, rank=None, **kwargs
) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = _countdown_ground_truth_for_index(i, kwargs)
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(
            1 for i in empty_indices if solution_str[i] == ground_truth[i]
        )
        return correct_cells / len(empty_indices)
    return 0.0


def _sudoku_fields_for_index(i: int, kwargs: dict):
    """Same batch vs single-row convention as countdown."""
    puzz = kwargs["puzzle"]
    sol = kwargs["solution"]
    if isinstance(puzz, (list, tuple)):
        puzzle_i = puzz[i]
    else:
        puzzle_i = puzz
    if isinstance(sol, (list, tuple)):
        ground_truth = sol[i]
    else:
        ground_truth = sol
    return puzzle_i, ground_truth


# For sudoku
def sudoku_reward_func(
    prompts, completions, run_name=None, step=None, rank=None, **kwargs
) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle, ground_truth = _sudoku_fields_for_index(i, kwargs)
        solution = extract_answer_sudoku(response)

        score = (
            0.0
            if solution is None
            else validate_sudoku_solution(solution, ground_truth, puzzle)
        )
        scores.append(score)

        if do_print:
            print("--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(
                f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})"
            )
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


# For math500
def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]

    # extracted_responses = []
    # answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    # for r in responses:
    #     try:
    #         r = remove_boxed(last_boxed_only_string(r))
    #     except:
    #         pass
    #     extracted_responses.append(r)

    extracted_responses = [extract_xml_answer(r) for r in responses]
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    # Only print for first example in per_device_batch_size
    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


# For math500
def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


# For b1
def block_format_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that encourages the usage of \block.
    Logic:
    - 0 blocks: 0.0
    - 1 block:  0.5
    - More blocks: Increases with diminishing returns (concave shape).
    - Caps at 10 blocks (approx 0.91 reward).
    Formula: reward = count / (count + 1)
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for r in responses:
        count = r.count("\\block")
        valid_count = min(count, 10)

        # f(0) = 0
        # f(1) = 1/2 = 0.5
        # f(2) = 2/3 ≈ 0.66
        # ...
        # f(10) = 10/11 ≈ 0.909
        reward = valid_count / (valid_count + 1.0)

        rewards.append(reward)

    return rewards


# ============== Reward Functions for MBPP/HumanEval ==============
def extract_code(completion: str, language: str = "python") -> str:
    """Extract code from markdown code blocks."""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else ""
    return extracted_answer


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Checks if the completion contains valid code in a code block.
    """

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content in completion_contents:
            # Extract code from between code blocks
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue

            # Get the first code block
            code = code_blocks[0].strip()

            # Check syntax if it's Python code
            if language == "python":
                try:
                    ast.parse(code)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
                rewards.append(
                    1.0 if syntax_valid else 0.5
                )  # grammar error get partial reward
            else:
                rewards.append(1.0)

        return rewards

    return code_format_reward


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets by running test cases.

    Assumes the dataset contains a `test_list` column with test cases.
    This is also used for code generation test like MBPP and HumanEval evaluation.
    """

    evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 30

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", f"{{code}}\\n{{case}}"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:
            continue

        passed += 1

    success_rate = (passed / total)
    return success_rate

code_snippet = {code_literal}
test_cases = {test_cases_literal}
rate = evaluate_code(code_snippet, test_cases)
print(rate)
"""
    # 1. compute format rewards
    format_rewards = get_code_format_reward(language="python")(completions)

    # 2. collect scripts and their indices
    template = evaluation_script_template
    scripts = []
    valid_indices = []
    code_list = []
    test_cases_list = []

    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            continue
        code = extract_code(completion[-1]["content"])
        code_list.append(code)
        tc = kwargs["test_list"][i]
        test_cases_list.append(tc)
        scripts.append(
            template.format(
                code_literal=repr(code),
                test_cases_literal=repr(tc),
            )
        )
        valid_indices.append(i)

    # 3. execute scripts
    # Outer timeout must cover inner `evaluate_code`: up to 30s per test case in the template.
    results = []
    for i, script in enumerate(scripts):
        try:
            tc = test_cases_list[i]
            n_cases = len(tc) if isinstance(tc, (list, tuple)) else 1
            outer_timeout = max(120, 30 * int(n_cases) + 30)
            process = subprocess.run(
                ["python3", "-c", script],
                text=True,
                capture_output=True,
                timeout=outer_timeout,
            )
            if process.returncode == 0:
                try:
                    results.append(float(process.stdout.strip()))
                except ValueError:
                    results.append(0.0)
            else:
                results.append(0.0)
        except Exception as e:
            print(f"Error executing code: {e}")
            results.append(0.0)

    # 4. fill results into a list of the same length as completions
    final_results = [0.0] * len(completions)
    for idx, res in zip(valid_indices, results):
        final_results[idx] = res

    # try:
    #     GREEN = "\033[92m"
    #     YELLOW = "\033[93m"
    #     BLUE = "\033[94m"
    #     RESET = "\033[0m"
    #     print(
    #         "*" * 100,
    #         f"\n{GREEN}Code:{RESET}\n{code_list[0]}\n",
    #         "-" * 20,
    #         f"\n{BLUE}Test Case:{RESET}\n{test_cases_list[0]}\n",
    #         "-" * 20,
    #         f"\n{YELLOW}Results:{RESET}\n{final_results}\n",
    #     )
    # except:
    #     pass

    return final_results


def code_reward_func(
    prompts, completions, run_name=None, step=None, **kwargs
) -> list[float]:
    """Wrapper reward function for MBPP/HumanEval code evaluation."""
    return code_reward(completions, **kwargs)


# ============== Reward Functions for Multiple Choice Tasks ==============


def extract_mc_answer(text: str) -> str:
    """Extract multiple choice answer (A, B, C, D, etc.) from model response."""
    import re

    # Try to find answer in <answer>X</answer> format
    answer_match = re.search(r"<answer>\s*([A-Ja-j])\s*</answer>", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Try to find "The answer is X" pattern
    answer_match = re.search(
        r"(?:the\s+)?answer\s+is\s*:?\s*([A-Ja-j])", text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).upper()

    # Try to find standalone letter at the end
    answer_match = re.search(
        r"(?:^|\s)([A-Ja-j])(?:\s*[.)]?\s*$)", text.strip()[-20:], re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).upper()

    # Try to find any letter followed by period or parenthesis
    answer_match = re.search(r"([A-Ja-j])\s*[.)]\s*$", text.strip(), re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    return ""


def mc_correctness_reward(completions, answer=None, **kwargs) -> list[float]:
    """
    Reward function for multiple choice questions.
    Returns 1.0 for correct answer, 0.0 for incorrect.
    """
    if answer is None:
        answer = kwargs.get("answer", [])

    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            content = completion[0]["content"] if completion else ""
        else:
            content = completion

        # Get ground truth answer
        if isinstance(answer, list):
            gt_answer = answer[i] if i < len(answer) else ""
        else:
            gt_answer = answer

        # Extract predicted answer
        pred_answer = extract_mc_answer(content)

        # Compare (case insensitive)
        if pred_answer.upper() == gt_answer.upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def mc_format_reward(completions, **kwargs) -> list[float]:
    """
    Format reward for multiple choice responses.
    Rewards proper use of <reasoning> and <answer> tags.
    """
    import re

    rewards = []

    for completion in completions:
        if isinstance(completion, list):
            content = completion[0]["content"] if completion else ""
        else:
            content = completion

        reward = 0.0

        # Check for reasoning tags
        if re.search(r"<reasoning>.*</reasoning>", content, re.DOTALL | re.IGNORECASE):
            reward += 0.3

        # Check for answer tags
        if re.search(r"<answer>\s*[A-Ja-j]\s*</answer>", content, re.IGNORECASE):
            reward += 0.7
        elif re.search(r"<answer>.*</answer>", content, re.DOTALL | re.IGNORECASE):
            reward += 0.3  # Has answer tag but wrong format

        rewards.append(reward)

    return rewards


def mc_reward_func(
    prompts, completions, run_name=None, step=None, **kwargs
) -> list[float]:
    """Combined reward function for multiple choice tasks."""
    format_rewards = mc_format_reward(completions, **kwargs)
    correctness_rewards = mc_correctness_reward(completions, **kwargs)

    # Weighted combination: 30% format + 70% correctness
    return [0.3 * f + 0.7 * c for f, c in zip(format_rewards, correctness_rewards)]


def _normalize_knk(s: str) -> str:
    if not s:
        return ""
    # normalize whitespace + lowercase; keep words for robust matching
    s = " ".join(s.split()).strip().lower()
    # remove lightweight punctuation that often differs in generations
    s = re.sub(r"[(){}\[\].,:;\"'`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_knk_answer_text(content: str) -> str:
    """Best-effort extraction of the final answer segment for K&K."""
    if not content:
        return ""

    # 1) Canonical tag format
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2) Common "Answer:" / "Final answer:" patterns
    match = re.search(
        r"(?:final\s+answer|answer)\s*[:：]\s*(.*)$",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # 3) Fallback: last chunk (models often conclude at the end)
    return content[-800:].strip()


_KNK_ROLE_PAT = re.compile(
    r"""
    (?:
        ^|\n|\r|\s
    )
    (?:\(\s*\d+\s*\)|\d+\s*[).:-])?
    \s*
    (?P<name>[A-Z][A-Za-z'\-]*)
    \s*
    (?:is|:|=)\s*
    (?:a|an)?\s*
    (?P<role>knight|knave)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_knk_assignments(text: str) -> dict[str, str]:
    """
    Parse lines like:
      (1) Zoey is a knave
      Oliver is a knight
      Zoey: knave
    Return {name_lower: role_lower}.
    """
    out: dict[str, str] = {}
    if not text:
        return out
    for m in _KNK_ROLE_PAT.finditer(text):
        name = (m.group("name") or "").strip().lower()
        role = (m.group("role") or "").strip().lower()
        if name and role in ("knight", "knave"):
            out[name] = role
    return out


def knights_knaves_reward_func(
    prompts, completions, run_name=None, step=None, **kwargs
) -> list[float]:
    """
    Reward for Knights-and-Knaves.

    Old behavior was strict exact match of <answer> vs solution_text_format, which
    often yields all-zeros due to minor formatting differences.

    New behavior:
    - Extract answer text (prefer <answer>, fallback to 'Answer:' or tail).
    - Parse per-person assignments (name -> knight/knave) for both pred & GT.
    - Reward is fractional accuracy over people; add a small format reward so
      well-formed outputs get non-zero signal early in training.
    """
    answer = kwargs.get("answer", [])
    rewards = []
    for i, completion in enumerate(completions):
        content = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        gt = answer[i] if i < len(answer) else ""

        pred_answer_text = _extract_knk_answer_text(content)
        gt_answer_text = gt or ""

        pred_map = _parse_knk_assignments(pred_answer_text)
        gt_map = _parse_knk_assignments(gt_answer_text)

        # correctness: structured compare when possible, else loose string match
        corr = 0.0
        if gt_map:
            if pred_map:
                keys = list(gt_map.keys())
                correct = sum(1 for k in keys if pred_map.get(k) == gt_map.get(k))
                corr = correct / max(1, len(keys))
            else:
                corr = 0.0
        else:
            pred_norm = _normalize_knk(pred_answer_text)
            gt_norm = _normalize_knk(gt_answer_text)
            corr = 1.0 if pred_norm and gt_norm and pred_norm == gt_norm else 0.0

        # format reward: encourage using <answer> and giving assignments
        fmt = 0.0
        if re.search(r"<answer>.*?</answer>", content, re.DOTALL | re.IGNORECASE):
            fmt += 0.15
        if pred_map:
            fmt += 0.15
        if re.search(r"<reasoning>.*?</reasoning>", content, re.DOTALL | re.IGNORECASE):
            fmt += 0.05
        # ultra-light fallback: even if parsing fails, mentioning the domain keywords
        # should yield a tiny non-zero signal (prevents all-zeros when formatting drifts).
        if ("knight" in (content or "").lower()) or ("knave" in (content or "").lower()):
            fmt += 0.05
        fmt = min(fmt, 0.3)

        rewards.append(min(1.0, 0.8 * corr + 0.2 * (fmt / 0.3 if fmt > 0 else 0.0)))
    return rewards


# --- Guru (LLM360/guru-RL-92k): vendored Reasoning360 default_compute_score ---
# Implementation lives in `rl/eval/guru/` (minimal Reasoning360 reward_score subset).


def _sanitize_guru_extra_info(extra: Any) -> dict:
    """HF/Arrow may yield JSON strings, Mapping proxies, or numpy scalars in extra_info."""
    if extra is None:
        return {}
    if isinstance(extra, str) and extra.strip():
        try:
            parsed = json.loads(extra)
        except json.JSONDecodeError:
            return {}
        extra = parsed
    if isinstance(extra, Mapping) and not isinstance(extra, dict):
        try:
            extra = dict(extra)
        except Exception:
            return {}
    if not isinstance(extra, dict):
        return {}
    out: dict = {}
    for k, v in extra.items():
        if (
            hasattr(v, "item")
            and callable(getattr(v, "item"))
            and not isinstance(v, (bytes, str))
        ):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        if isinstance(v, Mapping) and not isinstance(v, (str, bytes)):
            out[k] = _sanitize_guru_extra_info(v)
        else:
            out[k] = v
    return out


def _guru_question_text_from_prompt(prompt_sample: Any) -> str:
    """Best-effort user question text when extra_info may need ``question`` (legacy / diagnostics)."""
    if isinstance(prompt_sample, list):
        for m in reversed(prompt_sample):
            if isinstance(m, dict):
                role = str(m.get("role", "")).lower()
                if role in ("user", "human", "system"):
                    t = str(m.get("content", "")).strip()
                    if t:
                        return t
        if prompt_sample:
            last = prompt_sample[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
    return str(prompt_sample or "")


def _fill_guru_question_in_extra(extra: dict, prompt_sample: Optional[Any]) -> dict:
    """Ensure extra_info has ``question`` when callers expect it (e.g. future judges)."""
    if not isinstance(extra, dict):
        extra = {}
    q = extra.get("question")
    if isinstance(q, str) and q.strip():
        return extra
    text = _guru_question_text_from_prompt(prompt_sample)
    if not (isinstance(text, str) and text.strip()):
        return extra
    out = dict(extra)
    out["question"] = text
    return out


def _float_from_r360_score(res: Any) -> float:
    if isinstance(res, dict):
        if "score" in res:
            return float(res["score"])
        if "acc" in res:
            return float(res["acc"])
        return 0.0
    if isinstance(res, bool):
        return float(res)
    if isinstance(res, (int, float)):
        return float(res)
    try:
        return float(res[0])
    except Exception:
        return 0.0


def guru_unified_reward_func(
    prompts,
    completions,
    step=None,
    run_name=None,
    guru_data_source=None,
    guru_reward_ground_truth=None,
    guru_extra_info=None,
    **kwargs: Any,
) -> List[float]:
    """
    Per-sample rewards for the `guru` domain using the vendored Reasoning360 router
    ``rl.eval.guru.default_compute_score`` (same as ``NaiveRewardManager``:
    ``data_source``, full completion string, ``reward_model.ground_truth``,
    ``extra_info``).

    Dataset columns from `rl.eval.guru_dataset.load_guru_rl_train`:
      - guru_data_source
      - guru_reward_ground_truth
      - guru_extra_info

    Package ``rl/eval/guru/`` is vendored in-repo (no runtime ``verl`` import); install
    Python deps from Reasoning360's README if needed (sympy, pylatexenc, requests,
    etc.). Codegen uses ``coder1`` (env ``CODER1_EXEC``, default ``unsafe_local``).

    If ``data_source`` is not handled by the trimmed ``rl.eval.guru`` router,
    ``NotImplementedError`` is raised (not converted to 0.0), so training fails
    loudly instead of learning on bogus zero rewards.

    **Why mean reward can stay 0 for a long time (this is often expected):**
      - Many handlers (e.g. ``naive_dapo`` for ``math*`` and ``stem_web*``) only grade
        answers inside ``\\boxed{...}``; diffusion samples without that pattern score 0.
      - Wrong answers and failed code execution legitimately return 0.
    """
    from rl.eval.guru import default_compute_score as compute_score

    n = len(completions)
    prompts_seq: List[Any] = list(prompts) if isinstance(prompts, (list, tuple)) else []

    sandbox_url = os.environ.get("GURU_SANDBOX_FUSION_URL", "").strip() or None
    mem_mb = 1024
    mem_s = os.environ.get("GURU_SANDBOX_MEMORY_LIMIT_MB", "").strip()
    if mem_s.isdigit():
        mem_mb = int(mem_s)

    out: List[float] = []
    for i in range(n):
        ds_list = guru_data_source if isinstance(guru_data_source, (list, tuple)) else None
        gt_list = (
            guru_reward_ground_truth
            if isinstance(guru_reward_ground_truth, (list, tuple))
            else None
        )
        ex_list = guru_extra_info if isinstance(guru_extra_info, (list, tuple)) else None

        data_source = (
            str(ds_list[i]).strip()
            if ds_list is not None and i < len(ds_list) and ds_list[i] is not None
            else ""
        )
        ground_truth = (
            gt_list[i]
            if gt_list is not None and i < len(gt_list) and gt_list[i] is not None
            else ""
        )
        if isinstance(ground_truth, bytes):
            ground_truth = ground_truth.decode("utf-8", errors="replace")
        else:
            ground_truth = str(ground_truth)

        extra: dict = {}
        if ex_list is not None and i < len(ex_list) and ex_list[i] is not None:
            extra = _sanitize_guru_extra_info(ex_list[i])

        p_i = prompts_seq[i] if i < len(prompts_seq) else None
        rm = extra.get("reward_metric") if isinstance(extra, dict) else None
        if (
            isinstance(data_source, str)
            and (
                data_source.startswith("stem_web")
                or rm == "math_llm_judge"
            )
        ):
            extra = _fill_guru_question_in_extra(extra, p_i)

        comp = completions[i]
        solution_str = (
            comp[0]["content"]
            if comp and isinstance(comp, list) and isinstance(comp[0], dict)
            else str(comp)
        )

        if not data_source:
            warnings.warn(
                f"guru_unified_reward_func: empty guru_data_source at sample index {i} (reward=0). "
                "If mixing domains, check merged columns; pure-Guru rows should be filtered at load.",
                stacklevel=2,
            )
            out.append(0.0)
            continue

        try:
            res = compute_score(
                data_source,
                solution_str,
                ground_truth,
                extra_info=extra,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=None,
                memory_limit_mb=mem_mb,
            )
            out.append(_float_from_r360_score(res))
        except ImportError:
            raise
        except NotImplementedError:
            # Trimmed rl.eval.guru router: fail fast instead of training on fake zeros.
            raise
        except Exception as e:
            # Silent 0s hide misconfig (e.g. missing judge URL) and code bugs; log a few samples.
            if not hasattr(guru_unified_reward_func, "_score_exc_count"):
                guru_unified_reward_func._score_exc_count = 0  # type: ignore[attr-defined]
            c = guru_unified_reward_func._score_exc_count  # type: ignore[attr-defined]
            if c < 8:
                _logger.warning(
                    "guru_unified_reward_func: compute_score failed (data_source=%r, i=%s): %s",
                    data_source,
                    i,
                    e,
                    exc_info=c < 3,
                )
                guru_unified_reward_func._score_exc_count = c + 1  # type: ignore[attr-defined]
            out.append(0.0)

    return out
