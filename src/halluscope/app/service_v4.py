"""
Service layer for HalluScope V4.

Orchestrates text generation with purely mathematical hallucination detection.
No databases, no retrieval, no external knowledge bases.

V4 Signals:
1. Semantic Energy (distribution confidence intensity)
2. Entropy / Logprob Variance (uncertainty patterns)
3. Confidence-Stability Gap (CSG) - catches confident hallucinations
4. Logit Lens Divergence (LLD) - reasoning stability across layers
5. Representation Collapse Detection (RCD) - hallucination loops
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from halluscope.core.llm_backend_v4 import LLMBackend, GenerationResult
from halluscope.risk.hri import classify_risk


def _normalized_entropy(entropy: float, vocab_size: int) -> float:
    """Normalize entropy by maximum possible entropy."""
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log(vocab_size)
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def _get_clarification_questions(risk_level: str) -> List[str]:
    """Get clarification questions based on risk level."""
    if risk_level == "HIGH":
        return [
            "Can you verify this information with an external source?",
            "The model shows high confidence with semantic drift - please double-check facts.",
            "Would you like me to regenerate with a more constrained prompt?",
        ]
    elif risk_level == "MEDIUM":
        return [
            "Some claims may need verification.",
            "Consider cross-referencing specific details.",
        ]
    return []


def _generate_notes(result: GenerationResult) -> List[str]:
    """Generate human-readable notes about the generation."""
    notes = []
    
    # Check each V4 signal and add relevant notes
    if result.confidence_stability_gap > 0.4:
        notes.append(
            f"⚠️ High Confidence-Stability Gap ({result.confidence_stability_gap:.2f}): "
            f"Model is confident (conf={result.confidence:.2f}) but drifting from prompt semantics "
            f"(stability={result.semantic_stability:.2f}). This is a hallucination signal."
        )
    
    if result.logit_lens_divergence > 0.3:
        notes.append(
            f"⚠️ High Logit Lens Divergence ({result.logit_lens_divergence:.2f}): "
            f"Internal layer predictions are unstable, suggesting uncertain reasoning."
        )
    
    if result.representation_collapse > 0.5:
        notes.append(
            f"⚠️ Representation Collapse Detected ({result.representation_collapse:.2f}): "
            f"Model hidden states are converging, risk of repetitive/looping output."
        )
    
    if result.entropy_mean > 0.5:
        notes.append(
            f"ℹ️ High entropy ({result.entropy_mean:.2f}): "
            f"Model shows uncertainty in token selection."
        )
    
    if result.early_stopped:
        notes.insert(0, "⚠️ Generation stopped early due to high hallucination risk.")
    
    if not notes:
        notes.append("✅ No significant hallucination signals detected.")
    
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prompt(
    llm: LLMBackend,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.9,
    hri_threshold: float = 0.70,
    early_stop_enabled: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a prompt for hallucination risk using V4 mathematical signals.

    Pipeline:
    1. Generate response with per-token V4 signal monitoring
    2. Compute Semantic Energy, Entropy, CSG, LLD, RCD at each token
    3. Maintain running HRI throughout generation
    4. Early stop if HRI exceeds threshold
    5. Return complete signal breakdown for analysis

    Args:
        llm: Loaded LLMBackend V4 instance.
        prompt: User prompt string.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        hri_threshold: HRI value that triggers early stopping.
        early_stop_enabled: Whether to enable early stopping.
        debug: If True, print detailed signal breakdowns.

    Returns:
        Dict with all V4 signals, metrics, and diagnostics.
    """
    # Generate with V4 signal monitoring
    res = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        hri_threshold=hri_threshold,
        early_stop_enabled=early_stop_enabled,
        debug=debug,
    )
    
    # Extract final HRI and classify
    final_hri = res.hri_values[-1] if res.hri_values else 0.0
    hri_label = classify_risk(final_hri)
    
    # Compute normalized entropy
    entropy_norm = _normalized_entropy(res.entropy_mean, res.vocab_size)
    
    # Generate notes
    notes = _generate_notes(res)
    
    # Get clarification questions if risk is elevated
    clarifications = _get_clarification_questions(hri_label)
    
    # Extract just the generated text (not the prompt)
    generated_text = res.text[len(prompt):] if res.text.startswith(prompt) else res.text
    
    return {
        "answer": generated_text,
        
        # Token-level data for visualization
        "tokens": res.tokens,
        "token_risks": [round(r, 4) for r in res.token_risks],
        
        # HRI trajectory
        "hri_values": [round(h, 4) for h in res.hri_values],
        
        # Early stopping flag
        "early_stopped": res.early_stopped,
        
        # ─────────────────────────────────────────────────────────────────────
        # V4 SIGNAL BREAKDOWN (Purely Mathematical)
        # ─────────────────────────────────────────────────────────────────────
        
        "v4_signals": {
            # Distribution Geometry
            "entropy": round(res.entropy_mean, 4),
            "semantic_energy": round(res.semantic_energy_mean, 4),
            "logprob_variance": round(res.logprob_variance, 4),
            
            # Confidence-Stability Gap
            "confidence": round(res.confidence, 4),
            "semantic_stability": round(res.semantic_stability, 4),
            "confidence_stability_gap": round(res.confidence_stability_gap, 4),
            
            # Internal Representation Analysis
            "logit_lens_divergence": round(res.logit_lens_divergence, 4),
            "representation_collapse": round(res.representation_collapse, 4),
        },
        
        # Per-token V4 signals
        "per_token_signals": {
            "hri": [round(h, 4) for h in res.hri_values],
            "csg": [round(c, 4) for c in res.per_token_csg],
            "lld": [round(l, 4) for l in res.per_token_lld],
            "rcd": [round(r, 4) for r in res.per_token_rcd],
        },
        
        # HRI dynamics
        "hri_dynamics": {
            "slope": round(res.hri_slope, 4),
            "acceleration": round(res.hri_acceleration, 4),
        },
        
        # Risk breakdown (for UI charts)
        "risk_breakdown": {k: round(v, 4) for k, v in res.risk_breakdown.items()},
        
        # ─────────────────────────────────────────────────────────────────────
        # Aggregate metrics (backward compatible)
        # ─────────────────────────────────────────────────────────────────────
        
        "metrics": {
            "entropy_mean": round(res.entropy_mean, 4),
            "entropy_max": round(res.entropy_max, 4),
            "entropy_normalized": round(entropy_norm, 4),
            "entropy_variance": 0.0,  # Can be computed if needed
            "nll": round(res.nll, 4),
            "perplexity": round(res.perplexity, 4),
            "logprob_std": round(res.logprob_std, 4),
            "logprob_variance": round(res.logprob_variance, 4),
            "semantic_energy": round(res.semantic_energy_mean, 4),
            "kl_uniform_mean": None,  # Deprecated
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
        
        # Intervention message (if triggered)
        "intervention_message": res.intervention_message,
    }
