"""
Service layer: ties backend generation + HRI into one callable function.

TokenBurn-inspired orchestration - SIMPLIFIED for pre-hoc detection:
1. Generate text with real-time HRI monitoring via LLMBackend
2. Metrics are computed DURING generation (not after)
3. Early stopping enabled when HRI exceeds threshold

The heavy lifting is now done in LLMBackend.generate().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from halluscope.core.llm_backend import LLMBackend, GenerationResult
from halluscope.risk.hri import (
    classify_risk,
    explain,
    get_clarification_questions,
)


def _normalized_entropy(entropy: float, vocab_size: int) -> float:
    """
    Normalize entropy by maximum possible entropy.
    
    H_norm = H / log(vocab_size)
    """
    import math
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log(vocab_size)
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prompt(
    llm: LLMBackend,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.9,
    compute_kl: bool = True,  # Legacy parameter, ignored
    top_k_for_kl: int = 50,   # Legacy parameter, ignored
    hri_threshold: float = 0.70,
    early_stop_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a prompt for hallucination risk with PRE-HOC detection.

    TokenBurn-inspired pipeline (REAL-TIME):
      1. Generate response with per-token uncertainty monitoring
      2. Metrics computed DURING generation (entropy, NLL, std)
      3. Running HRI computed after each token
      4. Early stopping if HRI exceeds threshold
      5. Return token-level risks for UI highlighting

    Args:
        llm: Loaded LLMBackend instance.
        prompt: User prompt string.
        max_new_tokens: Max tokens to generate (default 60 for speed).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        hri_threshold: HRI value that triggers early stopping.
        early_stop_enabled: Whether to enable early stopping.

    Returns:
        Dict with: answer, metrics, hri, tokens, token_risks, hri_values,
                   early_stopped, notes, clarification_questions.
    """
    # Generate with real-time HRI monitoring
    res = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        hri_threshold=hri_threshold,
        early_stop_enabled=early_stop_enabled,
    )
    
    # Extract final HRI and classify
    final_hri = res.hri_values[-1] if res.hri_values else 0.0
    hri_label = classify_risk(final_hri)
    
    # Compute normalized entropy
    entropy_norm = _normalized_entropy(res.entropy_mean, res.vocab_size) if res.vocab_size > 0 else 0.0
    
    # Generate explanation notes
    notes = explain(
        entropy_mean=res.entropy_mean,
        nll=res.nll,
        ppl=res.perplexity,
        logprob_std=res.logprob_std,
        logprob_var=res.logprob_variance,
        kl_uniform=None,  # Not computed in new pipeline
    )
    
    # Add multi-signal risk notes
    if res.prompt_risk > 0.1:
        notes.insert(0, f"⚠️ High-risk prompt pattern detected (citation/specificity request)")
    
    if res.entity_risk > 0.1:
        notes.insert(0, f"⚠️ Novel entities in output not found in prompt: {', '.join(res.novel_entities[:3])}")
    
    if res.retrieval_risk > 0.08 and res.retrieval_analysis:
        for signal in res.retrieval_analysis.risk_signals[:2]:
            notes.insert(0, f"⚠️ {signal}")
    
    # Add early stopping note if triggered
    if res.early_stopped:
        notes.insert(0, "⚠️ Generation stopped early due to high hallucination risk.")
    
    # Get clarification questions if risk is elevated
    clarifications = res.clarification_questions
    if clarifications is None and hri_label in ("HIGH", "MEDIUM"):
        clarifications = get_clarification_questions(hri_label)
    
    # Extract just the generated text (not the prompt)
    generated_text = res.text[len(prompt):] if res.text.startswith(prompt) else res.text
    
    return {
        "answer": generated_text,
        
        # Token-level data for visualization
        "tokens": res.tokens,
        "token_risks": [round(r, 4) for r in res.token_risks],
        
        # HRI trajectory
        "hri_values": [round(h, 4) for h in res.hri_values],
        
        # Pre-hoc detection signals
        "early_stopped": res.early_stopped,
        
        # ─────────────────────────────────────────────────────────────────────
        # MULTI-SIGNAL RISK BREAKDOWN (NEW)
        # ─────────────────────────────────────────────────────────────────────
        
        "risk_signals": {
            "prompt_risk": round(res.prompt_risk, 4),
            "entity_risk": round(res.entity_risk, 4),
            "retrieval_risk": round(res.retrieval_risk, 4),
            "hri_slope": round(res.hri_slope, 4),
            "hri_acceleration": round(res.hri_acceleration, 4),
        },
        
        # Novel entities detected (potential fabrications)
        "novel_entities": res.novel_entities,
        
        # Retrieval risk signals
        "unverified_claims": res.retrieval_analysis.unverified_claims if res.retrieval_analysis else [],
        "risk_signals_detail": res.retrieval_analysis.risk_signals if res.retrieval_analysis else [],
        
        # Full risk breakdown for charts
        "risk_breakdown": {k: round(v, 4) for k, v in res.risk_breakdown.items()},
        
        # ─────────────────────────────────────────────────────────────────────
        # Aggregate metrics (backward compatible)
        # ─────────────────────────────────────────────────────────────────────
        
        "metrics": {
            "entropy_mean": round(res.entropy_mean, 4),
            "entropy_max": round(res.entropy_max, 4),
            "entropy_normalized": round(entropy_norm, 4),
            "entropy_variance": 0.0,  # Not computed per-step in new pipeline
            "nll": round(res.nll, 4),
            "perplexity": round(res.perplexity, 4),
            "logprob_std": round(res.logprob_std, 4),
            "logprob_variance": round(res.logprob_variance, 4),
            "kl_uniform_mean": None,  # Not computed in new pipeline
            "num_tokens": len(res.tokens),
        },
        
        # HRI score and label
        "hri": {
            "score": round(final_hri, 4),
            "label": hri_label,
        },
        
        # Explanation and clarification
        "notes": notes,
        "clarification_questions": clarifications,
    }
