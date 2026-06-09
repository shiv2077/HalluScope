"""
HalluScope – Confidence Metrics

Computes higher-order confidence signals from token log-probabilities.
These complement the raw log-prob statistics and capture *temporal*
patterns in model certainty across the generated sequence.
"""

from typing import List


def confidence_decay(logprobs: List[float], window: int = 5) -> float:
    """Measure how much model confidence drops from the beginning
    to the end of the generated sequence.

    Strategy:
        Compare the mean log-prob of the first `window` tokens to
        the mean of the last `window` tokens.  A positive value
        means the model grew *less* confident as it generated more
        text — a common hallucination signal.

    Args:
        logprobs: Sequence of per-token log-probabilities.
        window:   Number of tokens to consider at each end.

    Returns:
        A non-negative float representing the magnitude of the
        confidence drop.  Returns 0.0 when the sequence is too
        short for meaningful comparison.
    """
    if len(logprobs) < window * 2:
        # Not enough tokens; fall back to simple first-half / last-half
        if len(logprobs) < 2:
            return 0.0
        mid = len(logprobs) // 2
        first_half = logprobs[:mid]
        second_half = logprobs[mid:]
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)
        # A negative difference means confidence dropped (logprobs got
        # more negative).  We want the *magnitude* of the drop.
        return max(0.0, mean_first - mean_second)

    mean_start = sum(logprobs[:window]) / window
    mean_end = sum(logprobs[-window:]) / window
    return max(0.0, mean_start - mean_end)
