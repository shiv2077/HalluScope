"""
Token log-probability metrics for HalluScope.

TokenBurn-inspired metrics:
- token_logprobs: Per-token log-probability extraction
- aggregate_nll: Mean negative log-likelihood
- perplexity: exp(NLL) - interpretable confidence measure
- logprob_variance / logprob_std: Instability signals (hallucination indicator)

NLL measures how "surprising" the model's own outputs are to itself.
High NLL = model assigned low probability to the tokens it generated.
High variance = instability (confident filler + uncertain facts = hallucination).
"""

from __future__ import annotations

import math
from typing import List

import torch


def token_logprobs(
    scores: List[torch.Tensor],
    new_token_ids: torch.Tensor,
) -> List[float]:
    """
    Compute log-probability for each generated token.

    Args:
        scores: List of logit tensors, one per generated token.
                Each tensor has shape [batch=1, vocab_size].
        new_token_ids: Tensor of generated token IDs, shape [num_tokens].

    Returns:
        List of log-probability values (natural log) for each generated token.
    """
    logprobs: List[float] = []
    for logits, tok_id in zip(scores, new_token_ids):
        # Compute log-softmax and extract logprob for the chosen token
        log_probs = torch.log_softmax(logits[0], dim=-1)
        logp = log_probs[int(tok_id)].item()
        logprobs.append(float(logp))
    return logprobs


def aggregate_nll(token_logps: List[float]) -> float:
    """
    Aggregate token log-probabilities into mean NLL.

    NLL = -mean(log p(y_t))

    Args:
        token_logps: List of per-token log-probabilities (negative values).

    Returns:
        Mean negative log-likelihood. Higher = less confident.
    """
    if not token_logps:
        return 0.0
    mean_logp = sum(token_logps) / len(token_logps)
    return float(-mean_logp)


def perplexity(nll: float) -> float:
    """
    Compute perplexity from NLL.

    PPL = exp(NLL)

    TokenBurn interpretation:
    - PPL ≈ 1–20: Confident generation
    - PPL > 50: High uncertainty / guessing

    Args:
        nll: Mean negative log-likelihood.

    Returns:
        Perplexity score (≥ 1.0).
    """
    return math.exp(nll)


def logprob_variance(token_logps: List[float]) -> float:
    """
    Compute population variance of token log-probabilities.

    High variance indicates instability: the model alternates between
    confident tokens (filler words) and uncertain tokens (facts/names).
    This pattern is a strong hallucination signal (TokenBurn insight).

    Args:
        token_logps: List of per-token log-probabilities.

    Returns:
        Population variance of log-probabilities.
    """
    if len(token_logps) < 2:
        return 0.0
    mean = sum(token_logps) / len(token_logps)
    variance = sum((x - mean) ** 2 for x in token_logps) / len(token_logps)
    return float(variance)


def logprob_std(token_logps: List[float]) -> float:
    """
    Compute population standard deviation of token log-probabilities.

    This is the square root of logprob_variance, useful for
    combining with other metrics on similar scales.

    Args:
        token_logps: List of per-token log-probabilities.

    Returns:
        Population standard deviation of log-probabilities.
    """
    return math.sqrt(logprob_variance(token_logps))
