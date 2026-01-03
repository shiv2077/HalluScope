"""
Token-level entropy computation for HalluScope.

Entropy measures the "spread" of the probability distribution over the
vocabulary at each decoding step. High entropy = model is uncertain
about which token to generate next.

TokenBurn-style metrics:
- token_entropies: Per-token entropy from logits
- aggregate_entropy: Mean/max entropy across sequence
- normalized_entropy: Entropy / log(vocab_size) for comparability
- entropy_variance: Instability in uncertainty across tokens

Formula per step t:
    H_t = -sum_v( p(v|context) * log(p(v|context)) )

where v iterates over all vocabulary tokens.
"""

from __future__ import annotations

import math
from typing import List

import torch


def token_entropies(scores: List[torch.Tensor]) -> List[float]:
    """
    Compute token-level entropy from per-step logits.

    Uses exact entropy over full vocabulary (not top-k approximation).
    This is accurate since we have local access to all logits.

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


def normalized_entropy(entropy: float, vocab_size: int) -> float:
    """
    Normalize entropy by maximum possible entropy.

    H_norm = H / log(vocab_size)

    This gives a value in [0, 1] where:
    - 0 = deterministic (one token has probability 1)
    - 1 = uniform distribution (maximum uncertainty)

    Args:
        entropy: Entropy value in nats.
        vocab_size: Size of the vocabulary.

    Returns:
        Normalized entropy in [0, 1].
    """
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log(vocab_size)
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def entropy_variance(entropies: List[float]) -> float:
    """
    Compute population variance of token entropies.

    High variance indicates the model's uncertainty fluctuates across tokens.
    This can indicate hallucination: confident on filler words,
    uncertain on factual claims.

    Args:
        entropies: List of per-token entropy values.

    Returns:
        Population variance of entropies.
    """
    if len(entropies) < 2:
        return 0.0
    mean = sum(entropies) / len(entropies)
    variance = sum((x - mean) ** 2 for x in entropies) / len(entropies)
    return float(variance)
