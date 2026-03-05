"""
Service layer for HalluScope V5 — Three-Stage Hallucination Detection.

Orchestrates:
  Stage 1 — Pre-Hoc Prompt Risk (pattern analysis + prompt perplexity)
  Stage 2 — In-Generation Monitoring (token signals → generation HRI)
  Stage 3 — Post-Generation Verification (semantic analysis → verification risk)

Final HRI = max(
    BLEND_GEN * stage2_hri + BLEND_POST * verification_risk,
    verification_risk * POST_HOC_FLOOR
)

No databases, no retrieval, no external knowledge bases.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from halluscope.core.llm_backend_v4 import LLMBackend, GenerationResult
from halluscope.risk.hri import classify_risk
from halluscope.metrics.pre_hoc import (
    analyze_prompt_patterns,
    compute_prompt_risk,
    PreHocResult,
)
from halluscope.metrics.post_hoc import (
    verify as post_hoc_verify,
    PostHocResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Blending Constants
# ─────────────────────────────────────────────────────────────────────────────

BLEND_GEN = 0.55        # Weight for Stage 2 (generation HRI)
BLEND_POST = 0.45       # Weight for Stage 3 (verification risk)
POST_HOC_FLOOR = 0.80   # Post-hoc alone can push to HIGH: floor = risk * 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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
            "The model shows signs of confident fabrication — please double-check facts.",
            "Would you like me to regenerate with a more constrained prompt?",
        ]
    elif risk_level == "MEDIUM":
        return [
            "Some claims may need verification.",
            "Consider cross-referencing specific details.",
        ]
    return []


def _generate_notes(
    res: GenerationResult,
    pre_hoc: PreHocResult,
    post_hoc: PostHocResult,
) -> List[str]:
    """Generate human-readable notes from all three stages."""
    notes = []

    # Stage 1 notes
    if pre_hoc.prompt_risk_score > 0.35:
        details = []
        if pre_hoc.citation_score > 0.3:
            details.append("citation-style prompt")
        if pre_hoc.prompt_perplexity > 30:
            details.append(f"high prompt perplexity ({pre_hoc.prompt_perplexity:.0f})")
        if pre_hoc.temporal_risk > 0.2:
            details.append("temporal risk")
        if pre_hoc.entity_rarity > 0.3:
            details.append("rare/unknown entities")
        detail_str = ", ".join(details) if details else "elevated overall"
        notes.append(
            f"⚠️ Stage 1 — Risky Prompt ({pre_hoc.prompt_risk_score:.2f}): {detail_str}."
        )

    # Stage 2 notes
    if res.confidence_stability_gap > 0.4:
        notes.append(
            f"⚠️ Stage 2 — High CSG ({res.confidence_stability_gap:.2f}): "
            f"Model is confident but semantically drifting."
        )
    if res.logit_lens_divergence > 0.3:
        notes.append(
            f"⚠️ Stage 2 — High LLD ({res.logit_lens_divergence:.2f}): "
            f"Layer predictions are unstable."
        )
    if res.representation_collapse > 0.5:
        notes.append(
            f"⚠️ Stage 2 — Representation Collapse ({res.representation_collapse:.2f}): "
            f"Hidden states converging — risk of repetitive output."
        )

    # Stage 3 notes
    if post_hoc.confidence_paradox > 0.25:
        notes.append(
            f"⚠️ Stage 3 — Confidence Paradox ({post_hoc.confidence_paradox:.2f}): "
            f"Model is confident but producing novel content. Classic fabrication signal."
        )
    if post_hoc.semantic_novelty > 0.3:
        notes.append(
            f"⚠️ Stage 3 — Semantic Novelty ({post_hoc.semantic_novelty:.2f}): "
            f"Answer drifts significantly from prompt topic."
        )
    if post_hoc.claim_density > 0.3:
        notes.append(
            f"⚠️ Stage 3 — High Claim Density ({post_hoc.claim_density:.2f}): "
            f"Model makes many assertions without evidence."
        )
    if post_hoc.narrative_template_score > 0.3:
        notes.append(
            f"⚠️ Stage 3 — Fabrication Templates ({post_hoc.narrative_template_score:.2f}): "
            f"Output follows common hallucination narrative patterns."
        )

    if res.early_stopped:
        notes.insert(0, "⚠️ Generation stopped early due to high hallucination risk.")

    if not notes:
        notes.append("✅ No significant hallucination signals detected across all 3 stages.")

    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis — Three-Stage Pipeline
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
    V5 Three-Stage Hallucination Detection Pipeline.

    Stage 1: Pre-hoc prompt risk analysis (patterns + prompt perplexity)
    Stage 2: In-generation monitoring (token signals → generation HRI)
    Stage 3: Post-generation verification (semantic signals → verification risk)

    Final HRI blends all three stages.
    """

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Pre-Hoc Prompt Risk Analysis
    # ═════════════════════════════════════════════════════════════════════════

    pattern_result = analyze_prompt_patterns(prompt)

    if debug:
        print(f"\n[Stage 1] Pattern analysis:")
        print(f"  citation={pattern_result.citation_score:.3f} "
              f"entity_density={pattern_result.entity_density:.3f} "
              f"temporal={pattern_result.temporal_risk:.3f} "
              f"specificity={pattern_result.specificity_score:.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — In-Generation Monitoring
    # ═════════════════════════════════════════════════════════════════════════

    # Compute preliminary bias from pattern signals (before we have prompt ppl)
    preliminary_bias = (
        pattern_result.citation_score * 1.5
        + pattern_result.temporal_risk * 1.0
        + pattern_result.quoted_title_score * 1.0
    )

    res = llm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        hri_threshold=hri_threshold,
        early_stop_enabled=early_stop_enabled,
        prompt_risk_bias=preliminary_bias,
        debug=debug,
    )

    # Now compute full pre-hoc risk (with prompt perplexity from prefill)
    pre_hoc = compute_prompt_risk(
        pattern_result=pattern_result,
        prompt_perplexity=res.prompt_perplexity,
        min_token_logprob=res.min_prompt_logprob,
        low_prob_fraction=res.low_prob_fraction,
    )

    if debug:
        print(f"\n[Stage 1 FINAL] prompt_risk={pre_hoc.prompt_risk_score:.3f} "
              f"ppl={pre_hoc.prompt_perplexity:.1f} "
              f"rarity={pre_hoc.entity_rarity:.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Post-Generation Verification
    # ═════════════════════════════════════════════════════════════════════════

    generated_text = res.text[len(prompt):] if res.text.startswith(prompt) else res.text

    # Get embeddings for semantic novelty
    prompt_emb = None
    answer_emb = None
    try:
        prompt_emb = llm.embedder.encode(
            prompt, convert_to_tensor=True, normalize_embeddings=True,
        )
        if generated_text.strip():
            answer_emb = llm.embedder.encode(
                generated_text, convert_to_tensor=True, normalize_embeddings=True,
            )
    except Exception:
        pass

    post_hoc = post_hoc_verify(
        prompt=prompt,
        answer=generated_text,
        confidence=res.confidence,
        semantic_stability=res.semantic_stability,
        prompt_embedding=prompt_emb,
        answer_embedding=answer_emb,
    )

    if debug:
        print(f"\n[Stage 3] verification_risk={post_hoc.verification_risk:.3f}")
        print(f"  novelty={post_hoc.semantic_novelty:.3f} "
              f"claim={post_hoc.claim_density:.3f} "
              f"paradox={post_hoc.confidence_paradox:.3f} "
              f"template={post_hoc.narrative_template_score:.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL HRI — Three-Stage Blend
    # ═════════════════════════════════════════════════════════════════════════

    stage2_hri = res.hri_values[-1] if res.hri_values else 0.0

    # Weighted blend of generation + verification
    blended = BLEND_GEN * stage2_hri + BLEND_POST * post_hoc.verification_risk

    # Post-hoc alone can push risk high (catches confident hallucinations)
    post_floor = post_hoc.verification_risk * POST_HOC_FLOOR

    # Pre-hoc provides a minimum floor
    pre_floor = pre_hoc.prompt_risk_score * 0.5

    final_hri = max(blended, post_floor, pre_floor)
    final_hri = min(1.0, final_hri)

    hri_label = classify_risk(final_hri)

    if debug:
        print(f"\n[FINAL] stage2={stage2_hri:.3f} × {BLEND_GEN} = {BLEND_GEN*stage2_hri:.3f}")
        print(f"        post_hoc={post_hoc.verification_risk:.3f} × {BLEND_POST} = {BLEND_POST*post_hoc.verification_risk:.3f}")
        print(f"        blended={blended:.3f} post_floor={post_floor:.3f} pre_floor={pre_floor:.3f}")
        print(f"        FINAL HRI = {final_hri:.3f} → {hri_label}")

    # ═════════════════════════════════════════════════════════════════════════
    # Generate Notes & Clarification
    # ═════════════════════════════════════════════════════════════════════════

    notes = _generate_notes(res, pre_hoc, post_hoc)
    clarifications = _get_clarification_questions(hri_label)

    entropy_norm = _normalized_entropy(res.entropy_mean, res.vocab_size)

    return {
        "answer": generated_text,

        # Token-level data
        "tokens": res.tokens,
        "token_risks": [round(r, 4) for r in res.token_risks],

        # HRI trajectory (Stage 2 per-token)
        "hri_values": [round(h, 4) for h in res.hri_values],

        # Early stopping flag
        "early_stopped": res.early_stopped,

        # ─────────────────────────────────────────────────────────────────
        # THREE-STAGE BREAKDOWN
        # ─────────────────────────────────────────────────────────────────

        "stage1_prehoc": {
            "prompt_risk_score": round(pre_hoc.prompt_risk_score, 4),
            "citation_score": round(pre_hoc.citation_score, 4),
            "entity_density": round(pre_hoc.entity_density, 4),
            "temporal_risk": round(pre_hoc.temporal_risk, 4),
            "specificity_score": round(pre_hoc.specificity_score, 4),
            "prompt_perplexity": round(pre_hoc.prompt_perplexity, 2),
            "entity_rarity": round(pre_hoc.entity_rarity, 4),
            "quoted_title_score": round(pre_hoc.quoted_title_score, 4),
            "entities_found": pre_hoc.entities_found[:10],
            "years_found": pre_hoc.years_found,
        },

        "stage2_generation": {
            "generation_hri": round(stage2_hri, 4),
            "entropy": round(res.entropy_mean, 4),
            "semantic_energy": round(res.semantic_energy_mean, 4),
            "logprob_variance": round(res.logprob_variance, 4),
            "confidence": round(res.confidence, 4),
            "semantic_stability": round(res.semantic_stability, 4),
            "confidence_stability_gap": round(res.confidence_stability_gap, 4),
            "logit_lens_divergence": round(res.logit_lens_divergence, 4),
            "representation_collapse": round(res.representation_collapse, 4),
        },

        "stage3_posthoc": {
            "verification_risk": round(post_hoc.verification_risk, 4),
            "semantic_novelty": round(post_hoc.semantic_novelty, 4),
            "entity_expansion": round(post_hoc.entity_expansion, 4),
            "claim_density": round(post_hoc.claim_density, 4),
            "confidence_paradox": round(post_hoc.confidence_paradox, 4),
            "narrative_template_score": round(post_hoc.narrative_template_score, 4),
            "concept_explosion": round(post_hoc.concept_explosion, 4),
        },

        # V4-compatible signal breakdown
        "v4_signals": {
            "entropy": round(res.entropy_mean, 4),
            "semantic_energy": round(res.semantic_energy_mean, 4),
            "logprob_variance": round(res.logprob_variance, 4),
            "confidence": round(res.confidence, 4),
            "semantic_stability": round(res.semantic_stability, 4),
            "confidence_stability_gap": round(res.confidence_stability_gap, 4),
            "logit_lens_divergence": round(res.logit_lens_divergence, 4),
            "representation_collapse": round(res.representation_collapse, 4),
            "novelty": round(post_hoc.semantic_novelty, 4),
            "entity_expansion": round(post_hoc.entity_expansion, 4),
            "claim_density": round(post_hoc.claim_density, 4),
        },

        "per_token_signals": {
            "hri": [round(h, 4) for h in res.hri_values],
            "csg": [round(c, 4) for c in res.per_token_csg],
            "lld": [round(l, 4) for l in res.per_token_lld],
            "rcd": [round(r, 4) for r in res.per_token_rcd],
        },

        "hri_dynamics": {
            "slope": round(res.hri_slope, 4),
            "acceleration": round(res.hri_acceleration, 4),
        },

        "risk_breakdown": {k: round(v, 4) for k, v in res.risk_breakdown.items()},

        "metrics": {
            "entropy_mean": round(res.entropy_mean, 4),
            "entropy_max": round(res.entropy_max, 4),
            "entropy_normalized": round(entropy_norm, 4),
            "entropy_variance": 0.0,
            "nll": round(res.nll, 4),
            "perplexity": round(res.perplexity, 4),
            "logprob_std": round(res.logprob_std, 4),
            "logprob_variance": round(res.logprob_variance, 4),
            "semantic_energy": round(res.semantic_energy_mean, 4),
            "kl_uniform_mean": None,
            "num_tokens": len(res.tokens),
        },

        # Final HRI (three-stage blend)
        "hri": {
            "score": round(final_hri, 4),
            "label": hri_label,
            "stage2_hri": round(stage2_hri, 4),
            "verification_risk": round(post_hoc.verification_risk, 4),
            "prompt_risk": round(pre_hoc.prompt_risk_score, 4),
        },

        "notes": notes,
        "clarification_questions": clarifications,
        "intervention_message": res.intervention_message,
    }
