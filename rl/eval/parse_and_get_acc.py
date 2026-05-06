import json
import argparse
import re
import os
import sys
import glob
import tiktoken
from collections import defaultdict
from .parser_helper import is_equiv, remove_boxed, last_boxed_only_string

# Add parent directory to path for importing reward functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reward_func import code_reward, get_code_format_reward

_TIKTOKEN_ENC = None


def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    tokens = _TIKTOKEN_ENC.encode(text)
    return len(tokens)


def _safe_load_json_file(json_path: str):
    """
    Load a JSON file robustly.
    Handles empty files and cases where stdout/log text got mixed into the file by
    extracting the first top-level JSON object if possible.
    """
    try:
        with open(json_path, "r") as f:
            raw = f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read JSON file: {json_path}") from e

    if raw is None:
        raise ValueError(f"Empty JSON content: {json_path}")

    s = raw.strip()
    if not s:
        raise ValueError(f"Empty JSON content: {json_path}")

    # Fast path
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Recovery: extract the most likely JSON object region.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        return json.loads(candidate)

    # Give up with a more helpful error.
    raise ValueError(f"Invalid JSON content (not parseable): {json_path}")


def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if (
                    boxed_content
                    and boxed_content != "..."
                    and not re.match(r"^\.+$", boxed_content)
                ):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(
                r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
            )
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(
                r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
            )
            if answer_match:
                parsed_answer = answer_match.group(1).strip()
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def parse_countdown_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0

    processed_items = []

    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(generated_text)
        total_effective_tokens += effective_tokens

        # Extract available numbers and target from ground_truth
        numbers = []
        target = None

        if isinstance(ground_truth, list) and len(ground_truth) == 2:
            numbers = ground_truth[0]
            target = ground_truth[1]
        else:
            # Fallback to parsing from question if ground_truth is not in expected format
            numbers_match = re.search(
                r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE
            )
            if numbers_match:
                numbers_str = numbers_match.group(1)
                numbers = [int(num.strip()) for num in numbers_str.split(",")]

            target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
            if target_match:
                target = int(target_match.group(1))

        equation = ""
        try:
            equation = remove_boxed(last_boxed_only_string(generated_text))
        except:
            # Try to extract from answer tags
            answer_match = re.search(
                r"<answer>(.*?)</answer>", generated_text, re.DOTALL
            )
            if answer_match:
                equation = answer_match.group(1).strip()
            else:
                equation = generated_text

        # Replace LaTeX operators with Python operators
        equation = (
            equation.replace(r"\div", "/")
            .replace(r"\times", "*")
            .replace(r"\cdot", "*")
        )

        # Check for equation with equals sign and extract only the expression part
        equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
        if equation_match:
            equation = equation_match.group(1).strip()

        is_correct = False
        result = None

        # Validate and evaluate the equation
        is_valid = validate_equation(equation, numbers)
        if is_valid:
            result = evaluate_equation(equation)
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                total_correct += 1

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def parse_sudoku_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct_cells = total_empty_cells = total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Extract puzzle
        puzzle_str = ""
        if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
            puzzle_str = question[:16]
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)

        # Extract solution using regex patterns
        solution_str = ""
        patterns = [
            r"<answer>.*?```\s*([\d\s]+)```",
            r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
            r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
            r".*?(\d{16})\s*</answer>",
            r"\b(\d{16})\b",
        ]

        for pattern in patterns:
            if solution_str:
                break
            match = re.search(pattern, raw_generation, re.DOTALL)
            if match and match.group(1).strip():
                solution_str = match.group(1).strip()

        solution_str = re.sub(r"\s", "", solution_str)

        # Handle solution length
        if not solution_str:
            correct_cells = 0
        else:
            if len(solution_str) < 16:
                solution_str = solution_str + "0" * (16 - len(solution_str))
            elif len(solution_str) > 16:
                solution_str = solution_str[:16]
            correct_cells = sum(
                1 for i in empty_indices if solution_str[i] == ground_truth[i]
            )

        accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
        total_correct_cells += correct_cells
        total_empty_cells += empty_cells

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": solution_str,
                "ground_truth": ground_truth,
                "empty_cells": empty_cells,
                "correct_cells": correct_cells,
                "accuracy": accuracy,
                "effective_tokens": effective_tokens,
            }
        )
    return (
        total_correct_cells,
        total_empty_cells,
        processed_items,
        total_effective_tokens * 8,
    )


def parse_code_answers(json_path=None, json_data=None):
    """Parse code answers for MBPP and HumanEval datasets."""
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    # Extract id from filename for logging
    id_value = 0
    if json_path:
        s_match = re.search(r"_(\d+)_generations\.json$", json_path)
        if s_match:
            id_value = int(s_match.group(1))

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    format_reward_func = get_code_format_reward(language="python")

    for idx, item in enumerate(data.get("generations", [])):
        total_processed += 1

        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")
        test_list = item.get("test_list", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Evaluate code using code_reward function
        test_completions = [[{"content": raw_generation}]]
        reward_kwargs = {"test_list": [test_list]}
        current_reward = code_reward(test_completions, **reward_kwargs)[0] * 1.0

        is_correct = 1 if current_reward == 1 else 0
        total_correct += is_correct
        # print(
        #     f"{id_value}/{idx}/{len(data['generations'])} current_reward {current_reward:.3f} correct {is_correct}"
        # )

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def _normalize_knk_answer(s: str) -> str:
    """Normalize knights-and-knaves answer for comparison."""
    if not s:
        return ""
    return " ".join(s.split()).strip()


def extract_knk_answer(response: str) -> str:
    """Extract content from <answer> tag for knights-and-knaves."""
    if not response:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return _normalize_knk_answer(match.group(1))
    return ""


# Structured parsing for Knights-and-Knaves answers (order/whitespace-robust).
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
    """Parse 'Name is a knight/knave' lines into {name_lower: role_lower}."""
    out: dict[str, str] = {}
    if not text:
        return out
    for m in _KNK_ROLE_PAT.finditer(text):
        name = (m.group("name") or "").strip().lower()
        role = (m.group("role") or "").strip().lower()
        if name and role in ("knight", "knave"):
            out[name] = role
    return out


def parse_knights_knaves_answers(json_path=None, json_data=None):
    """
    Parse knights-and-knaves with a relaxed, structured metric.

    Old behavior used strict exact match of <answer> vs solution_text_format, which is
    often overly harsh and yields near-zero accuracy even for partially correct outputs.

    New behavior:
    - Extract <answer> content.
    - Parse per-person assignments (name -> knight/knave) for both pred & GT.
    - Score is fractional accuracy over GT people (in [0,1]).

    Note: We still keep a boolean `is_correct` for \"all people correct\".
    """
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0.0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed = extract_knk_answer(raw_generation)

        pred_map = _parse_knk_assignments(parsed)
        gt_map = _parse_knk_assignments(ground_truth or "")

        score = 0.0
        if gt_map:
            if pred_map:
                keys = list(gt_map.keys())
                correct = sum(1 for k in keys if pred_map.get(k) == gt_map.get(k))
                score = correct / max(1, len(keys))
            else:
                score = 0.0
        else:
            # Fallback: if GT isn't parseable, revert to exact normalized match.
            gt_norm = _normalize_knk_answer(ground_truth)
            score = 1.0 if parsed and gt_norm and parsed == gt_norm else 0.0

        is_correct = bool(score == 1.0)
        total_correct += float(score)

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "score": float(score),
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def extract_mc_answer(response: str) -> str:
    """Extract multiple choice answer (A/B/C/D/...) from response."""
    if not response:
        return ""

    # Try to find answer in <answer> tags first
    answer_match = re.search(
        r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Extract letter from answer text
        letter_match = re.search(r"\b([A-J])\b", answer_text, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

    # Common patterns: "answer is A", "A)", "[A]", "(A)", "A."
    patterns = [
        r"(?:answer|correct|choice)(?:\s+is)?[:\s]*\[?([A-J])\]?",
        r"\b([A-J])\s*(?:is\s+)?(?:correct|the\s+answer)",
        r"(?:^|\s)\(([A-J])\)",
        r"(?:^|\s)\[([A-J])\]",
        r"(?:^|\s)([A-J])\.",
        r"(?:^|\s)([A-J])\)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find any standalone letter A-J at the end
    match = re.search(r"\b([A-J])\b(?:\s*[.\)]?\s*)?$", response.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def parse_mc_answers(json_path=None, json_data=None):
    """Parse multiple choice answers for MMLU, MMLU-Pro, HellaSwag, ARC-C datasets."""
    if json_path:
        data = _safe_load_json_file(json_path)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1

        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")  # Should be a letter A/B/C/D/etc.
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        # Extract answer from generation
        parsed_answer = extract_mc_answer(raw_generation)

        # Compare with ground truth (case-insensitive)
        is_correct = parsed_answer.upper() == str(ground_truth).upper()
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def aggregate_results(directory="."):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))

    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "accuracy": 0.0,
            "questions": [],
            "total_effective_tokens": 0,
        }
    )

    for json_file in json_files:
        filename = os.path.basename(json_file)
        setup_name = extract_setup_name(filename)

        if setup_name:
            # print(f"Processing {filename}...")
            try:
                if "gsm" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_gsm_answers(json_path=json_file)
                elif "math" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_math_answers(json_path=json_file)
                elif "countdown" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_countdown_answers(json_path=json_file)
                elif "sudoku" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_sudoku_answers(json_path=json_file)
                elif any(
                    name in setup_name for name in ["mbpp", "humaneval", "kodcode"]
                ):
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_code_answers(json_path=json_file)
                elif any(
                    name in setup_name
                    for name in [
                        "mmlu",
                        "mmlu_pro",
                        "hellaswag",
                        "arc_c",
                        "arc_e",
                        "gpqa",
                    ]
                ):
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_mc_answers(json_path=json_file)
                elif "knights_and_knaves" in setup_name:
                    (
                        correct,
                        processed,
                        detailed_results,
                        total_effective_tokens,
                    ) = parse_knights_knaves_answers(json_path=json_file)
                else:
                    continue
            except Exception as e:
                print(f"[WARN] Skipping invalid result file: {json_file}")
                print(f"       Reason: {type(e).__name__}: {e}")
                continue

            setups[setup_name]["correct"] += correct
            setups[setup_name]["processed"] += processed
            setups[setup_name]["total_effective_tokens"] += total_effective_tokens
            setups[setup_name]["questions"].extend(detailed_results)

    # Calculate final accuracy and save results
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100
            if results["processed"] > 0
            else 0
        )
        results["avg_effective_tokens"] = (
            results["total_effective_tokens"] / results["processed"]
            if len(results["questions"]) > 0
            else 0
        )
    # Header
    header_format = "{:<40} {:>12} {:>25}"
    print(
        header_format.format(
            "Setup (task_model_seqlen_diffusteps)", "Accuracy", "Avg Effective Tokens"
        )
    )
    print("-" * 80)

    # Data rows
    row_format = "{:<40} {:>11.2f}% {:>25.2f}"
    for setup, results in sorted(setups.items()):
        print(
            row_format.format(
                setup, results["accuracy"], results["avg_effective_tokens"]
            )
        )

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="eval_results",
        help="Directory containing evaluation results",
    )
    args = parser.parse_args()

    aggregate_results(directory=args.directory)
    # aggregate_results(directory="eval/eval_baselines")
    # aggregate_results(directory="eval_results")
