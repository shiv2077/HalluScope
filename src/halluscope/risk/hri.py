"""
Hallucination Risk Index (HRI) for HalluScope.

TokenBurn-inspired composite risk estimation.

HRI-v1.1 combines multiple uncertainty signals:
- Entropy: How uncertain is the model about token choices?
- NLL: How confident was the model in selected tokens?
- Logprob std: Instability (confident filler + uncertain facts = hallucination)
- Optional KL: Distance from uniform distribution

Formula:
    HRI = sigmoid(wH*entropy + wN*nll + wV*logprob_std + wK*kl + b)

Risk levels:
- LOW: score < 0.45
- MEDIUM: 0.45 <= score < 0.70
- HIGH: score >= 0.70
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Default Weights (can be overridden via configs/risk.yaml)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "wH": 0.35,  # Entropy weight
    "wN": 0.35,  # NLL weight
    "wV": 0.30,  # Logprob std weight
    "wK": 0.0,   # KL weight (optional, disabled by default)
    "b": -2.0,   # Bias term
}

# Risk thresholds
THRESHOLD_LOW = 0.45
THRESHOLD_HIGH = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def compute_hri(
    entropy_mean: float,
    nll: float,
    logprob_std: float = 0.0,
    kl_uniform: Optional[float] = None,
    wH: float = DEFAULT_WEIGHTS["wH"],
    wN: float = DEFAULT_WEIGHTS["wN"],
    wV: float = DEFAULT_WEIGHTS["wV"],
    wK: float = DEFAULT_WEIGHTS["wK"],
    b: float = DEFAULT_WEIGHTS["b"],
) -> float:
    """
    Compute Hallucination Risk Index v1.1 (TokenBurn-inspired).

    HRI = sigmoid(wH*entropy + wN*nll + wV*logprob_std + wK*kl + b)

    The bias term shifts typical safe outputs below 0.5.

    Args:
        entropy_mean: Mean token entropy (nats). Higher = more uncertain.
        nll: Mean negative log-likelihood. Higher = less confident.
        logprob_std: Standard deviation of log-probs. Higher = more unstable.
        kl_uniform: Optional KL divergence from uniform. Lower = more uncertain.
        wH: Weight for entropy term.
        wN: Weight for NLL term.
        wV: Weight for logprob_std term.
        wK: Weight for KL term (inverted: low KL = add risk).
        b: Bias term (shifts sigmoid curve).

    Returns:
        HRI score in [0, 1]. Higher = higher hallucination risk.
    """
    x = wH * entropy_mean + wN * nll + wV * logprob_std + b
    
    # KL contribution: low KL means uncertain, so we invert
    # If KL is provided and wK > 0, subtract scaled KL (low KL = higher risk)
    if kl_uniform is not None and wK != 0:
        # Normalize KL to reasonable scale (empirical: typical KL is 0-10)
        x -= wK * kl_uniform  # Subtract: low KL increases x, high KL decreases x
    
    return sigmoid(x)


def classify_risk(score: float) -> str:
    """
    Classify HRI score into risk level.

    Args:
        score: HRI score in [0, 1].

    Returns:
        Risk level: "LOW", "MEDIUM", or "HIGH".
    """
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    if score >= THRESHOLD_LOW:
        return "MEDIUM"
    return "LOW"


def compute_hri_with_label(
    entropy_mean: float,
    nll: float,
    logprob_std: float = 0.0,
    kl_uniform: Optional[float] = None,
    **kwargs,
) -> Tuple[float, str]:
    """
    Compute HRI score and risk label together.

    Args:
        entropy_mean: Mean token entropy.
        nll: Mean negative log-likelihood.
        logprob_std: Standard deviation of log-probs.
        kl_uniform: Optional KL divergence from uniform.
        **kwargs: Additional weight parameters.

    Returns:
        Tuple of (score, label).
    """
    score = compute_hri(entropy_mean, nll, logprob_std, kl_uniform, **kwargs)
    label = classify_risk(score)
    return score, label


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Generation
# ─────────────────────────────────────────────────────────────────────────────

def explain(
    entropy_mean: float,
    nll: float,
    ppl: float,
    logprob_std: float,
    logprob_var: float,
    kl_uniform: Optional[float] = None,
) -> List[str]:
    """
    Generate human-readable explanation notes for HRI.

    TokenBurn-style: Explain each signal's contribution to risk.

    Args:
        entropy_mean: Mean token entropy.
        nll: Mean negative log-likelihood.
        ppl: Perplexity (exp(nll)).
        logprob_std: Standard deviation of log-probs.
        logprob_var: Variance of log-probs.
        kl_uniform: Optional KL divergence from uniform.

    Returns:
        List of explanation strings.
    """
    notes = []

    # Entropy observations
    if entropy_mean > 5.0:
        notes.append(
            "⚠️ Very high entropy: The model was highly uncertain about word choices."
        )
    elif entropy_mean > 4.0:
        notes.append(
            "⚡ High entropy: Notable uncertainty in next-token prediction."
        )
    elif entropy_mean > 3.0:
        notes.append(
            "📊 Moderate entropy: Some uncertainty detected."
        )

    # Perplexity observations (TokenBurn-style)
    if ppl > 50:
        notes.append(
            f"🎲 High perplexity ({ppl:.1f}): The model may be guessing."
        )
    elif ppl > 20:
        notes.append(
            f"📈 Elevated perplexity ({ppl:.1f}): Confidence is below typical levels."
        )

    # NLL observations
    if nll > 4.0:
        notes.append(
            "❌ Very high NLL: The model assigned low probability to its own outputs."
        )
    elif nll > 3.0:
        notes.append(
            "⚡ High NLL: The model was not highly confident in its token choices."
        )

    # Instability observations (TokenBurn key insight)
    if logprob_std > 2.0:
        notes.append(
            "🌊 High instability: Log-probability fluctuates significantly across tokens. "
            "This often indicates confident filler words mixed with uncertain factual claims."
        )
    elif logprob_std > 1.0:
        notes.append(
            "📉 Moderate instability: Some variation in token confidence detected."
        )

    # KL observations (optional)
    if kl_uniform is not None:
        if kl_uniform < 2.0:
            notes.append(
                "🎯 Low KL divergence: Distribution is close to uniform (high uncertainty)."
            )
        elif kl_uniform > 8.0:
            notes.append(
                "✅ High KL divergence: Distribution is peaked (model is confident)."
            )

    return notes or ["✅ Uncertainty signals are within normal bounds."]


# ─────────────────────────────────────────────────────────────────────────────
# Clarification Questions
# ─────────────────────────────────────────────────────────────────────────────

def clarification_questions_high() -> List[str]:
    """
    Return clarification questions for HIGH risk.

    These questions help users provide more specific context,
    which may reduce hallucination risk on a follow-up query.
    """
    return [
        "Can you specify the exact entity, date, or location the answer must be grounded on?",
        "Do you want a strictly factual answer (must be verifiable) or a conceptual explanation?",
        "What source or context should the response rely on (notes, paper, dataset, etc.)?",
    ]


def clarification_questions_medium() -> List[str]:
    """
    Return a softer clarification question for MEDIUM risk.

    Less intrusive than HIGH risk questions.
    """
    return [
        "Would you like me to provide sources or caveats for this answer?",
    ]


def get_clarification_questions(label: str) -> Optional[List[str]]:
    """
    Get appropriate clarification questions based on risk label.

    Args:
        label: Risk label ("LOW", "MEDIUM", "HIGH").

    Returns:
        List of clarification questions, or None for LOW risk.
    """
    if label == "HIGH":
        return clarification_questions_high()
    if label == "MEDIUM":
        return clarification_questions_medium()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Legacy aliases for backwards compatibility
# ─────────────────────────────────────────────────────────────────────────────

def compute_hri_v1(
    entropy_mean: float,
    nll: float,
    wH: float = 0.5,
    wN: float = 0.5,
    b: float = -2.0,
) -> float:
    """Legacy v1 computation (entropy + NLL only)."""
    return compute_hri(entropy_mean, nll, 0.0, None, wH, wN, 0.0, 0.0, b)


def explain_v1(entropy_mean: float, nll: float) -> List[str]:
    """Legacy v1 explanation (entropy + NLL only)."""
    ppl = math.exp(nll) if nll < 100 else float("inf")
    return explain(entropy_mean, nll, ppl, 0.0, 0.0, None)
