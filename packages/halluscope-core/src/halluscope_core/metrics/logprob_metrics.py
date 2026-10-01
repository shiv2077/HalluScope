"""
HalluScope – Log-Probability Metrics

Pure-function module that derives statistical metrics from the
per-token log-probability sequence returned by any LLM adapter.

These metrics feed into the HRI (Hallucination Risk Index) engine.
"""

import math
from typing import List


def mean_logprob(logprobs: List[float]) -> float:
    """Arithmetic mean of the token log-probabilities.

    A lower (more negative) mean indicates that the model was, on
    average, less confident in its token choices.

    Returns 0.0 for an empty sequence to avoid division-by-zero.
    """
    if not logprobs:
        return 0.0
    return sum(logprobs) / len(logprobs)


def logprob_variance(logprobs: List[float]) -> float:
    """Population variance of the token log-probabilities.

    Higher variance signals inconsistent model confidence across
    the generated sequence, which often correlates with hallucination.

    Returns 0.0 for sequences shorter than 2 tokens.
    """
    if len(logprobs) < 2:
        return 0.0
    mean = sum(logprobs) / len(logprobs)
    return sum((lp - mean) ** 2 for lp in logprobs) / len(logprobs)


def logprob_std(logprobs: List[float]) -> float:
    """Population standard deviation of the token log-probabilities.

    Square root of logprob_variance – provided as a convenience since
    standard deviation is more interpretable in the same unit as the
    original values.
    """
    return math.sqrt(logprob_variance(logprobs))
