# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Guru-only subset of Reasoning360 verl/utils/reward_score (Apache-2.0).
# Lives under rl/eval/guru/ next to other eval code. Trimmed: livebench, ifeval, gsm8k, search_r1, prime_code, …


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Route by data_source for LLM360/guru-RL-92k-style rows (Reasoning360 semantics)."""
    reward_metric = None
    if extra_info and isinstance(extra_info, dict):
        reward_metric = extra_info.get("reward_metric", None)

    if data_source.startswith("math"):
        if reward_metric == "prime_math":
            from . import prime_math

            res = prime_math.compute_score(solution_str, ground_truth)
        elif reward_metric == "math_llm_judge":
            # Local-only Guru policy: skip external LLM judge (OpenAI/vLLM).
            from . import naive_dapo

            res = naive_dapo.compute_score(
                solution_str, ground_truth, extra_info=extra_info or {}
            )
        else:
            from . import naive_dapo

            res = naive_dapo.compute_score(
                solution_str, ground_truth, extra_info=extra_info or {}
            )
    elif data_source.startswith("codegen"):
        from . import coder1

        res = coder1.compute_score(
            solution_str, ground_truth, extra_info=extra_info or {}
        )
    elif data_source.startswith("simulation__codeio"):
        from . import codeio

        res = codeio.compute_score(solution_str, ground_truth)
    elif data_source.startswith("simulation__cruxeval"):
        from . import cruxeval

        res = cruxeval.compute_score(solution_str, ground_truth)
    elif data_source.startswith("simulation__arcagi") or data_source.startswith(
        "simulation__barc"
    ):
        from . import arcagi

        res = arcagi.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__zebra_puzzle"):
        from . import zebra_puzzle

        res = zebra_puzzle.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__ordering_puzzle"):
        from . import puzzles_dataset

        res = puzzles_dataset.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__graph"):
        from . import graph_dataset

        res = graph_dataset.compute_score(solution_str, ground_truth)
    elif data_source.startswith("table"):
        from . import tablereason

        res = tablereason.compute_score(solution_str, ground_truth)
    elif data_source.startswith("stem__gpqa"):
        from . import gpqa
        from . import supergpqa

        if "no_box" in data_source:
            res = gpqa.compute_score(solution_str, ground_truth)
        else:
            res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith("stem__supergpqa"):
        from . import supergpqa

        res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith("stem_web"):
        # Local-only: same \boxed + sympy/string grading as DAPO math (no STEM_LLM_JUDGE_URL).
        from . import naive_dapo

        res = naive_dapo.compute_score(
            solution_str, ground_truth, extra_info=extra_info or {}
        )
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in (
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ):
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(
            f"Guru reward: no handler for {data_source=} (trimmed package)."
        )

    if isinstance(res, dict):
        return res
    if isinstance(res, (bool, int, float)):
        return float(res)
    return float(res[0])


__all__ = ["default_compute_score"]
