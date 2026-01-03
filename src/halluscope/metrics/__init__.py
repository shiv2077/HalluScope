"""
Metrics module for HalluScope.

TokenBurn-inspired uncertainty metrics for hallucination detection:
- Entropy: Token-level and normalized entropy
- Log-probability: NLL, perplexity, variance/std
- KL Divergence: Distance from uniform distribution
"""

from .entropy import (
    token_entropies,
    aggregate_entropy,
    normalized_entropy,
    entropy_variance,
)
from .logprob import (
    token_logprobs,
    aggregate_nll,
    perplexity,
    logprob_variance,
    logprob_std,
)
from .kl_divergence import kl_to_uniform

__all__ = [
    # Entropy
    "token_entropies",
    "aggregate_entropy",
    "normalized_entropy",
    "entropy_variance",
    # Logprob
    "token_logprobs",
    "aggregate_nll",
    "perplexity",
    "logprob_variance",
    "logprob_std",
    # KL
    "kl_to_uniform",
]

