"""
KL Divergence metrics for HalluScope.

TokenBurn-inspired KL divergence computation using top-k approximation.
KL(p || uniform) measures how far the model's distribution is from
maximum uncertainty (uniform over vocabulary).

Low KL = model is uncertain (close to uniform)
High KL = model is confident (peaked distribution)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


def kl_to_uniform_step(topk_probs: torch.Tensor, vocab_size: int) -> float:
    """
    Compute KL divergence from model distribution to uniform for one step.

    Uses top-k approximation:
    KL(p||u) ≈ sum_{i in topk} p_i * log(p_i * vocab_size)
             + tail_mass * log(tail_mass * vocab_size / (vocab_size - k))

    where tail_mass = max(0, 1 - sum(topk_probs)).

    Args:
        topk_probs: Top-k probabilities for one step, shape [k].
        vocab_size: Total vocabulary size.

    Returns:
        KL divergence in nats for this step.
    """
    if vocab_size <= 0:
        return 0.0
    
    k = len(topk_probs)
    if k == 0:
        return 0.0
    
    # Uniform probability
    u = 1.0 / vocab_size
    
    # Sum of top-k probabilities
    topk_sum = topk_probs.sum().item()
    tail_mass = max(0.0, 1.0 - topk_sum)
    
    # KL contribution from top-k tokens
    # KL = sum_i p_i * log(p_i / u) = sum_i p_i * (log(p_i) + log(vocab_size))
    kl = 0.0
    for p_i in topk_probs:
        p = p_i.item()
        if p > 1e-12:
            kl += p * math.log(p / u)
    
    # KL contribution from tail (remaining tokens)
    if tail_mass > 1e-12 and vocab_size > k:
        # Assume tail mass is distributed uniformly over remaining tokens
        tail_tokens = vocab_size - k
        u_tail = tail_mass / tail_tokens
        # Each tail token contributes: u_tail * log(u_tail / u)
        # Total: tail_tokens * u_tail * log(u_tail / u) = tail_mass * log(u_tail / u)
        kl += tail_mass * math.log(u_tail / u)
    
    return max(0.0, kl)


def kl_to_uniform(
    topk_probs: Optional[List[torch.Tensor]],
    vocab_size: int,
) -> Optional[float]:
    """
    Compute mean KL divergence from model distribution to uniform.

    This measures how peaked (confident) the model's distribution is.
    Low KL ≈ uncertain (close to uniform).
    High KL ≈ confident (peaked on few tokens).

    Args:
        topk_probs: List of top-k probability tensors, one per step.
                   Each tensor has shape [k].
                   If None or empty, returns None.
        vocab_size: Total vocabulary size.

    Returns:
        Mean KL divergence across all steps, or None if not available.
    """
    if topk_probs is None or len(topk_probs) == 0:
        return None
    
    if vocab_size <= 0:
        return None
    
    kl_values = [kl_to_uniform_step(probs, vocab_size) for probs in topk_probs]
    
    if not kl_values:
        return None
    
    return sum(kl_values) / len(kl_values)
