"""
Metrics module for HalluScope.

Multi-signal uncertainty metrics for hallucination detection:
- Entropy: Token-level and normalized entropy
- Log-probability: NLL, perplexity, variance/std
- KL Divergence: Distance from uniform distribution
- Self-Consistency: Multi-sample agreement (v3.0)
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

# NEW v3.0: Self-consistency module
from .self_consistency import (
    compute_self_consistency,
    quick_consistency_check,
    ConsistencyAnalysis,
)

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
    # Self-consistency
    "compute_self_consistency",
    "quick_consistency_check",
    "ConsistencyAnalysis",
]

