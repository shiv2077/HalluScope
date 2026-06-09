"""Verbalized Confidence baseline (Lin et al., 2022 / Tian et al., 2023).

Prompt the model to answer AND state a confidence in [0, 1]; risk is
1 - confidence. The LLM call is injected so the confidence parser is testable
without a model.
"""

from __future__ import annotations

import re
from typing import Callable

_PCT = re.compile(r"confidence[^0-9]*(\d{1,3})\s*%", re.IGNORECASE)
_DEC = re.compile(r"confidence[^0-9]*(0?\.\d+|1\.0+|\d{1,3})", re.IGNORECASE)

VERBALIZED_PROMPT = (
    "Answer the question, then on a new line state your confidence that your "
    "answer is correct as 'Confidence: X' where X is between 0 and 1.\n\n"
    "Question: {question}"
)


def parse_confidence(text: str) -> float:
    """Extract a confidence ∈ [0, 1] from a model response."""
    if not text:
        return 0.5
    m = _PCT.search(text)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 100.0))
    m = _DEC.search(text)
    if m:
        val = float(m.group(1))
        if val > 1.0:  # e.g. "Confidence: 80" meaning 80%
            val /= 100.0
        return max(0.0, min(1.0, val))
    return 0.5  # could not parse => neutral


class VerbalizedConfidenceScorer:
    """Risk = 1 - verbalized confidence."""

    name = "baseline_verbalized_confidence"

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn

    def score(self, question: str) -> float:
        return 1.0 - parse_confidence(
            self.llm_fn(VERBALIZED_PROMPT.format(question=question))
        )
