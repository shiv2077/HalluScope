"""
Metrics module for HalluScope.

This module contains uncertainty metrics for hallucination detection,
including entropy, log-probability, disagreement, and calibration metrics.
"""

from .entropy import token_entropies, aggregate_entropy
from .logprob import token_logprobs, aggregate_nll

__all__ = [
    "token_entropies",
    "aggregate_entropy",
    "token_logprobs",
    "aggregate_nll",
]

