"""P(True) baseline (Kadavath et al., 2022).

Ask the model whether its own answer is true; map that to P(true). Risk is
1 - P(true). The LLM call is injected (``llm_fn``: prompt -> text), so the
parsing is unit-testable without a model.
"""

from __future__ import annotations

import re
from typing import Callable

_PROB = re.compile(r"(?:prob(?:ability)?[^0-9]*)(\d?\.\d+|\d+%)", re.IGNORECASE)
_DECIMAL = re.compile(r"\b(0?\.\d+|1\.0+)\b")

P_TRUE_PROMPT = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n\n"
    "Is the proposed answer true? Respond with the probability it is true as a "
    "number between 0 and 1 (e.g. 'Probability true: 0.82'), or just 'True'/'False'."
)


def parse_p_true(text: str) -> float:
    """Extract P(true) ∈ [0, 1] from a model response."""
    if not text:
        return 0.5
    m = _PROB.search(text)
    if m:
        tok = m.group(1)
        val = float(tok[:-1]) / 100.0 if tok.endswith("%") else float(tok)
        return max(0.0, min(1.0, val))
    m = _DECIMAL.search(text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    low = text.strip().lower()
    if low.startswith("true") or "is true" in low:
        return 1.0
    if low.startswith("false") or "is false" in low or "incorrect" in low:
        return 0.0
    return 0.5  # could not parse => neutral


class PTrueScorer:
    """Risk = 1 - P(true)."""

    name = "baseline_p_true"

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def score(self, question: str, answer: str) -> float:
        prompt = P_TRUE_PROMPT.format(question=question, answer=answer)
        return 1.0 - parse_p_true(self.llm_fn(prompt))
