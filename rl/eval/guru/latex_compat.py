"""
Centralized LaTeX-to-text for Guru scorers (naive_dapo, prime_math, math_llm_judge).

Raises a clear ImportError if ``pylatexenc`` is missing (common on fresh clusters).
"""
from __future__ import annotations

from typing import Any, Optional

_L2T_INSTANCE: Optional[Any] = None


def latex_nodes_to_text(expr: str) -> str:
    """Convert LaTeX string to plain text using pylatexenc (cached)."""
    global _L2T_INSTANCE
    if _L2T_INSTANCE is None:
        try:
            from pylatexenc import latex2text

            _L2T_INSTANCE = latex2text.LatexNodes2Text()
        except ImportError as e:
            raise ImportError(
                "Guru math scoring requires the 'pylatexenc' package.\n"
                "  pip install pylatexenc\n"
                "or from the dLLM-R1 repo root:\n"
                "  pip install -r requirements-guru.txt\n"
                "(sympy is also listed there.)"
            ) from e
    return _L2T_INSTANCE.latex_to_text(expr)
