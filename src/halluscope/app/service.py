"""
Service layer: ties backend generation + metrics + HRI into one callable function.

This module orchestrates the full inference pipeline:
1. Generate text with logits via LLMBackend
2. Compute entropy and NLL from per-step logits
3. Compute HRI-v1 and produce explanation + clarification
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from halluscope.core.llm_backend import LLMBackend
from halluscope.metrics.entropy import token_entropies, aggregate_entropy
from halluscope.metrics.logprob import token_logprobs, aggregate_nll
from halluscope.risk.hri import compute_hri_v1, explain_v1, clarification_questions


# ─────────────────────────────────────────────────────────────────────────────
# Risk Level Thresholds
# ─────────────────────────────────────────────────────────────────────────────

HRI_THRESHOLD_HIGH = 0.65
HRI_THRESHOLD_MEDIUM = 0.40


def classify_risk(hri: float) -> str:
    """Classify HRI score into risk level."""
    if hri >= HRI_THRESHOLD_HIGH:
        return "high"
    if hri >= HRI_THRESHOLD_MEDIUM:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prompt(
    llm: LLMBackend,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Analyze a prompt for hallucination risk.

    Steps:
      1. Generate response with per-step logits (output_scores=True).
      2. Compute token-level entropies and aggregate to entropy_mean.
      3. Compute token-level log-probs and aggregate to NLL.
      4. Compute HRI-v1 = sigmoid(wH*entropy + wN*nll + b).
      5. Generate explanation notes and clarification questions if high risk.

    Args:
        llm: Loaded LLMBackend instance.
        prompt: User prompt string.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        Dict with keys: answer, metrics, hri, risk_level, notes, clarification_questions.
    """
    # 1. Generate with logits
    res = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # 2. Compute entropy from logits
    entropies = token_entropies(res.scores)
    entropy_mean = aggregate_entropy(entropies, mode="mean")

    # 3. Compute NLL from logits + chosen tokens
    logprobs = token_logprobs(res.scores, res.new_token_ids)
    nll = aggregate_nll(logprobs)

    # 4. Compute HRI-v1
    hri = compute_hri_v1(entropy_mean, nll)
    risk_level = classify_risk(hri)

    # 5. Explanation and clarification
    notes = explain_v1(entropy_mean, nll)
    clarifications: Optional[List[str]] = None
    if hri >= HRI_THRESHOLD_HIGH:
        clarifications = clarification_questions()

    return {
        "answer": res.text,
        "metrics": {
            "entropy_mean": round(entropy_mean, 4),
            "nll": round(nll, 4),
            # Sprint-2/3 placeholders
            "disagreement": None,
            "edfl_proxy": None,
        },
        "hri": round(hri, 4),
        "risk_level": risk_level,
        "notes": notes,
        "clarification_questions": clarifications,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sprint-2 TODO: Multi-sample Disagreement
# ─────────────────────────────────────────────────────────────────────────────
# def compute_disagreement(llm, prompt, K=5):
#     """Generate K samples and compute semantic disagreement."""
#     samples = [llm.generate(prompt, temperature=1.0).text for _ in range(K)]
#     # Use sentence-transformers to embed, then compute pairwise cosine similarity
#     # Disagreement = 1 - mean(similarity)
#     return disagreement_score, samples


# ─────────────────────────────────────────────────────────────────────────────
# Sprint-3 TODO: EDFL Proxy (Evidence Sensitivity)
# ─────────────────────────────────────────────────────────────────────────────
# def compute_edfl_proxy(llm, prompt, answer):
#     """Measure how much the answer changes when context is perturbed."""
#     # Perturb prompt, regenerate, compare semantic drift
#     pass
