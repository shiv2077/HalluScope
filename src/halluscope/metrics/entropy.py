"""
Token-level entropy computation for HalluScope.

Entropy is computed from the model's per-step logits and aggregated
(mean by default) to form a sequence-level uncertainty signal.

Entropy measures the "spread" of the probability distribution over the
vocabulary at each decoding step. High entropy = model is uncertain
about which token to generate next.

Formula per step t:
    H_t = -sum_v( p(v|context) * log(p(v|context)) )

where v iterates over all vocabulary tokens.
"""

from __future__ import annotations

from typing import List

import torch


def token_entropies(scores: List[torch.Tensor]) -> List[float]:
    """
    Compute token-level entropy from per-step logits.

    Args:
        scores: List of logit tensors, one per generated token.
                Each tensor has shape [batch=1, vocab_size].

    Returns:
        List of entropy values (in nats) for each generated token.
    """
    entropies: List[float] = []
    for logits in scores:
        # logits shape: [1, vocab_size]
        probs = torch.softmax(logits[0], dim=-1)
        # Entropy: H = -sum(p * log(p)), with small epsilon for numerical stability
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        entropies.append(float(entropy))
    return entropies


def aggregate_entropy(entropies: List[float], mode: str = "mean") -> float:
    """
    Aggregate token-level entropies to a single sequence-level value.

    Args:
        entropies: List of per-token entropy values.
        mode: Aggregation mode - "mean" or "max".

    Returns:
        Aggregated entropy value.
    """
    if not entropies:
        return 0.0
    if mode == "mean":
        return float(sum(entropies) / len(entropies))
    if mode == "max":
        return float(max(entropies))
    raise ValueError(f"Unknown aggregation mode: {mode}. Use 'mean' or 'max'.")
