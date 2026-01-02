"""
Token log-probability metrics for HalluScope.

Computes per-token log p(y_t | context) from logits and the chosen token id.
Aggregates into mean negative log-likelihood (NLL) as a confidence signal.

NLL measures how "surprising" the model's own outputs are to itself.
High NLL = model assigned low probability to the tokens it generated,
which may indicate the model is generating less reliable content.

Formula:
    NLL = -mean_t( log p(y_t | y_{<t}, x) )

where y_t is the token generated at step t.
"""

from __future__ import annotations

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
        List of log-probability values for each generated token.
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
