"""
Service layer: ties backend generation + metrics + HRI into one callable function.

TokenBurn-inspired orchestration:
1. Generate text with logits via LLMBackend
2. Compute entropy, NLL, perplexity, variance/std from per-step logits
3. Optionally compute KL divergence from top-k probabilities
4. Compute HRI-v1.1 and produce explanation + clarification
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from halluscope.core.llm_backend import LLMBackend, GenerationResult
from halluscope.metrics.entropy import (
    token_entropies,
    aggregate_entropy,
    normalized_entropy,
    entropy_variance,
)
from halluscope.metrics.logprob import (
    token_logprobs,
    aggregate_nll,
    perplexity,
    logprob_variance,
    logprob_std,
)
from halluscope.metrics.kl_divergence import kl_to_uniform
from halluscope.risk.hri import (
    compute_hri,
    classify_risk,
    explain,
    get_clarification_questions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prompt(
    llm: LLMBackend,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    compute_kl: bool = True,
    top_k_for_kl: int = 50,
) -> Dict[str, Any]:
    """
    Analyze a prompt for hallucination risk.

    TokenBurn-inspired pipeline:
      1. Generate response with per-step logits (output_scores=True).
      2. Compute token-level log-probs → NLL, perplexity, variance, std.
      3. Compute token-level entropies → entropy_mean, normalized, variance.
      4. Optionally compute KL divergence from top-k probabilities.
      5. Compute HRI = sigmoid(wH*entropy + wN*nll + wV*std + b).
      6. Generate explanation notes and clarification questions.

    Args:
        llm: Loaded LLMBackend instance.
        prompt: User prompt string.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        compute_kl: Whether to compute KL divergence (requires top-k probs).
        top_k_for_kl: Top-k value for KL computation (default 50).

    Returns:
        Dict with keys: answer, metrics, hri, notes, clarification_questions.
    """
    # 1. Generate with logits (and optionally top-k probs for KL)
    res = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k_probs=top_k_for_kl if compute_kl else None,
    )

    # 2. Compute log-probability metrics
    token_logps = token_logprobs(res.scores, res.new_token_ids)
    nll_val = aggregate_nll(token_logps)
    ppl_val = perplexity(nll_val)
    logprob_var = logprob_variance(token_logps)
    logprob_std_val = logprob_std(token_logps)

    # 3. Compute entropy metrics
    entropies = token_entropies(res.scores)
    entropy_mean = aggregate_entropy(entropies, mode="mean")
    entropy_max = aggregate_entropy(entropies, mode="max")
    entropy_var = entropy_variance(entropies)
    entropy_norm = normalized_entropy(entropy_mean, res.vocab_size)

    # 4. Compute KL divergence (optional)
    kl_uniform_mean: Optional[float] = None
    if compute_kl and res.topk_probs is not None:
        kl_uniform_mean = kl_to_uniform(res.topk_probs, res.vocab_size)

    # 5. Compute HRI-v1.1
    hri_score = compute_hri(
        entropy_mean=entropy_mean,
        nll=nll_val,
        logprob_std=logprob_std_val,
        kl_uniform=kl_uniform_mean,
    )
    hri_label = classify_risk(hri_score)

    # 6. Explanation and clarification
    notes = explain(
        entropy_mean=entropy_mean,
        nll=nll_val,
        ppl=ppl_val,
        logprob_std=logprob_std_val,
        logprob_var=logprob_var,
        kl_uniform=kl_uniform_mean,
    )
    clarifications = get_clarification_questions(hri_label)

    return {
        "answer": res.text,
        "metrics": {
            "entropy_mean": round(entropy_mean, 4),
            "entropy_max": round(entropy_max, 4),
            "entropy_normalized": round(entropy_norm, 4),
            "entropy_variance": round(entropy_var, 4),
            "nll": round(nll_val, 4),
            "perplexity": round(ppl_val, 4),
            "logprob_std": round(logprob_std_val, 4),
            "logprob_variance": round(logprob_var, 4),
            "kl_uniform_mean": round(kl_uniform_mean, 4) if kl_uniform_mean is not None else None,
            "num_tokens": len(token_logps),
        },
        "hri": {
            "score": round(hri_score, 4),
            "label": hri_label,
        },
        "notes": notes,
        "clarification_questions": clarifications,
    }
