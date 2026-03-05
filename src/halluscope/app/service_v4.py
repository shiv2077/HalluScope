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


# ─────────────────────────────────────────────────────────────────────────────
# Token Risk Classification & Visualization (V5.1)
# ─────────────────────────────────────────────────────────────────────────────

def classify_token_risk(hri: float) -> str:
    """Classify a single token's risk level based on HRI value."""
    if hri >= 0.7:
        return "HIGH"
    elif hri >= 0.4:
        return "MEDIUM"
    return "LOW"


def build_token_diagnostics(
    tokens: List[str],
    hri_values: List[float],
    entropy_values: List[float],
    energy_values: List[float],
    logprob_values: List[float],
    per_token_csg: List[float],
    per_token_lld: List[float],
    per_token_rcd: List[float],
) -> List[Dict[str, Any]]:
    """Build per-token diagnostic records with risk classification."""
    diagnostics = []
    n = len(tokens)
    for i in range(n):
        hri = hri_values[i] if i < len(hri_values) else 0.0
        diagnostics.append({
            "index": i,
            "token": tokens[i],
            "logprob": round(logprob_values[i], 4) if i < len(logprob_values) else 0.0,
            "entropy": round(entropy_values[i], 4) if i < len(entropy_values) else 0.0,
            "energy": round(energy_values[i], 4) if i < len(energy_values) else 0.0,
            "csg": round(per_token_csg[i], 4) if i < len(per_token_csg) else 0.0,
            "lld": round(per_token_lld[i], 4) if i < len(per_token_lld) else 0.0,
            "rcd": round(per_token_rcd[i], 4) if i < len(per_token_rcd) else 0.0,
            "hri": round(hri, 4),
            "risk_level": classify_token_risk(hri),
        })
    return diagnostics


def compute_hallucination_segments(
    token_diagnostics: List[Dict[str, Any]],
    min_risk: str = "MEDIUM",
) -> List[Dict[str, Any]]:
    """
    Group consecutive high-risk tokens into hallucination segments.
    
    Args:
        token_diagnostics: Per-token diagnostic records
        min_risk: Minimum risk level to consider ("MEDIUM" or "HIGH")
    
    Returns:
        List of segment dicts with start/end indices and average metrics
    """
    risk_levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    min_level = risk_levels.get(min_risk, 1)
    
    segments = []
    current_segment = None
    
    for diag in token_diagnostics:
        token_level = risk_levels.get(diag["risk_level"], 0)
        
        if token_level >= min_level:
            if current_segment is None:
                current_segment = {
                    "start_token": diag["index"],
                    "end_token": diag["index"],
                    "tokens": [diag["token"]],
                    "hri_values": [diag["hri"]],
                    "entropy_values": [diag["entropy"]],
                    "csg_values": [diag["csg"]],
                    "lld_values": [diag["lld"]],
                    "rcd_values": [diag["rcd"]],
                    "energy_values": [diag["energy"]],
                }
            else:
                current_segment["end_token"] = diag["index"]
                current_segment["tokens"].append(diag["token"])
                current_segment["hri_values"].append(diag["hri"])
                current_segment["entropy_values"].append(diag["entropy"])
                current_segment["csg_values"].append(diag["csg"])
                current_segment["lld_values"].append(diag["lld"])
                current_segment["rcd_values"].append(diag["rcd"])
                current_segment["energy_values"].append(diag["energy"])
        else:
            if current_segment is not None:
                # Finalize segment
                segments.append(_finalize_segment(current_segment))
                current_segment = None
    
    # Handle trailing segment
    if current_segment is not None:
        segments.append(_finalize_segment(current_segment))
    
    return segments


def _finalize_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute averages for a hallucination segment."""
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0
    
    return {
        "start_token": seg["start_token"],
        "end_token": seg["end_token"],
        "text": "".join(seg["tokens"]),
        "num_tokens": len(seg["tokens"]),
        "avg_hri": avg(seg["hri_values"]),
        "max_hri": round(max(seg["hri_values"]), 4) if seg["hri_values"] else 0.0,
        "signal_breakdown": {
            "entropy": avg(seg["entropy_values"]),
            "csg": avg(seg["csg_values"]),
            "lld": avg(seg["lld_values"]),
            "rcd": avg(seg["rcd_values"]),
            "energy": avg(seg["energy_values"]),
        }
    }


def highlight_hallucinations_html(
    tokens: List[str],
    hri_values: List[float],
) -> str:
    """
    Generate HTML with color-coded tokens based on risk level.
    
    Colors:
        LOW (HRI < 0.4)    → green
        MEDIUM (0.4-0.7)   → orange
        HIGH (HRI ≥ 0.7)   → red
    """
    parts = []
    for i, tok in enumerate(tokens):
        hri = hri_values[i] if i < len(hri_values) else 0.0
        risk = classify_token_risk(hri)
        
        if risk == "HIGH":
            color = "#dc3545"  # red
            bg = "rgba(220, 53, 69, 0.2)"
        elif risk == "MEDIUM":
            color = "#fd7e14"  # orange
            bg = "rgba(253, 126, 20, 0.2)"
        else:
            color = "#28a745"  # green
            bg = "rgba(40, 167, 69, 0.1)"
        
        # Escape HTML characters
        safe_tok = (
            tok.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
               .replace('"', "&quot;")
        )
        
        parts.append(
            f'<span style="color:{color};background:{bg};padding:1px 3px;'
            f'border-radius:3px;cursor:pointer" '
            f'title="Token {i}: HRI={hri:.3f} ({risk})">{safe_tok}</span>'
        )
    
    return "".join(parts)


def highlight_hallucinations_markdown(
    tokens: List[str],
    hri_values: List[float],
) -> str:
    """
    Generate markdown-style highlighted text (for plain-text contexts).
    
    Format:
        LOW tokens → plain
        MEDIUM tokens → **bold**
        HIGH tokens → ***bold italic*** or ~~strikethrough~~
    """
    parts = []
    for i, tok in enumerate(tokens):
        hri = hri_values[i] if i < len(hri_values) else 0.0
        risk = classify_token_risk(hri)
        
        if risk == "HIGH":
            parts.append(f"«{tok}»")  # Mark high-risk with guillemets
        elif risk == "MEDIUM":
            parts.append(f"[{tok}]")  # Mark medium-risk with brackets
        else:
            parts.append(tok)
    
    return "".join(parts)


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
    early_stop_enabled: bool = False,  # V5.1: Disabled by default for visualization mode
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

        # ─────────────────────────────────────────────────────────────────
        # V5.1 VISUALIZATION & HALLUCINATION ANALYSIS
        # ─────────────────────────────────────────────────────────────────

        # Per-token diagnostics with risk classification
        "token_diagnostics": build_token_diagnostics(
            tokens=res.tokens,
            hri_values=res.hri_values,
            entropy_values=res.entropy_values,
            energy_values=res.energy_values,
            logprob_values=res.logprob_values,
            per_token_csg=res.per_token_csg,
            per_token_lld=res.per_token_lld,
            per_token_rcd=res.per_token_rcd,
        ),

        # Hallucination segments (consecutive high-risk tokens)
        "hallucination_segments": compute_hallucination_segments(
            build_token_diagnostics(
                tokens=res.tokens,
                hri_values=res.hri_values,
                entropy_values=res.entropy_values,
                energy_values=res.energy_values,
                logprob_values=res.logprob_values,
                per_token_csg=res.per_token_csg,
                per_token_lld=res.per_token_lld,
                per_token_rcd=res.per_token_rcd,
            ),
            min_risk="MEDIUM",
        ),

        # Highlighted HTML output (color-coded tokens)
        "highlighted_html": highlight_hallucinations_html(
            tokens=res.tokens,
            hri_values=res.hri_values,
        ),

        # Plain-text marked output
        "highlighted_text": highlight_hallucinations_markdown(
            tokens=res.tokens,
            hri_values=res.hri_values,
        ),

        # Global HRI statistics
        "hri_stats": {
            "mean": round(sum(res.hri_values) / len(res.hri_values), 4) if res.hri_values else 0.0,
            "peak": round(max(res.hri_values), 4) if res.hri_values else 0.0,
            "min": round(min(res.hri_values), 4) if res.hri_values else 0.0,
            "final": round(res.hri_values[-1], 4) if res.hri_values else 0.0,
        },

        # Per-token entropy/energy/lld for visualization charts
        "per_token_entropy": [round(e, 4) for e in res.entropy_values],
        "per_token_energy": [round(e, 4) for e in res.energy_values],
        "per_token_logprob": [round(lp, 4) for lp in res.logprob_values],
    }
