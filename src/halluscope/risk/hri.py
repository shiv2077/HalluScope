"""
Hallucination Risk Index (HRI) for HalluScope.

Sprint-1: HRI-v1 uses only entropy_mean and NLL (confidence collapse).
Outputs a bounded score in [0,1] plus a lightweight explanation.

Future versions (Sprint-2, Sprint-3) will incorporate:
- Multi-sample disagreement (semantic similarity across K samples)
- EDFL proxy (evidence sensitivity / context perturbation)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core HRI Computation
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def compute_hri_v1(
    entropy_mean: float,
    nll: float,
    wH: float = 0.5,
    wN: float = 0.5,
    b: float = -2.0,
) -> float:
    """
    Compute Hallucination Risk Index v1.

    HRI = sigmoid(wH * entropy_mean + wN * nll + b)

    The bias term (b) shifts typical safe outputs below 0.5.
    Calibrate weights on TruthfulQA/HaluEval for production use.

    Args:
        entropy_mean: Mean token entropy (nats). Higher = more uncertain.
        nll: Mean negative log-likelihood. Higher = less confident.
        wH: Weight for entropy term.
        wN: Weight for NLL term.
        b: Bias term (shifts sigmoid curve).

    Returns:
        HRI score in [0, 1]. Higher = higher hallucination risk.
    """
    x = wH * entropy_mean + wN * nll + b
    return sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint-2 Placeholder: HRI-v2 with Disagreement
# ─────────────────────────────────────────────────────────────────────────────

def compute_hri_v2(
    entropy_mean: float,
    nll: float,
    disagreement: float,
    wH: float = 0.35,
    wN: float = 0.35,
    wD: float = 0.30,
    b: float = -2.0,
) -> float:
    """
    (Sprint-2) HRI with multi-sample disagreement.

    TODO: Implement after adding disagreement metric.
    """
    x = wH * entropy_mean + wN * nll + wD * disagreement + b
    return sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint-3 Placeholder: HRI-v3 with EDFL Proxy
# ─────────────────────────────────────────────────────────────────────────────

def compute_hri_v3(
    entropy_mean: float,
    nll: float,
    disagreement: float,
    edfl_proxy: float,
    weights: Optional[Dict[str, float]] = None,
    b: float = -2.0,
) -> float:
    """
    (Sprint-3) Full HRI with all signals.

    TODO: Implement after adding EDFL proxy metric.
    """
    w = weights or {"H": 0.25, "N": 0.25, "D": 0.25, "E": 0.25}
    x = (
        w["H"] * entropy_mean
        + w["N"] * nll
        + w["D"] * disagreement
        + w["E"] * edfl_proxy
        + b
    )
    return sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Generation
# ─────────────────────────────────────────────────────────────────────────────

def explain_v1(entropy_mean: float, nll: float) -> List[str]:
    """
    Generate human-readable explanation notes for HRI-v1.

    Args:
        entropy_mean: Mean token entropy.
        nll: Mean negative log-likelihood.

    Returns:
        List of explanation strings.
    """
    notes = []

    # Entropy thresholds (empirical, adjust after calibration)
    if entropy_mean > 5.0:
        notes.append(
            "Very high token entropy: the model is highly uncertain about word choices."
        )
    elif entropy_mean > 4.0:
        notes.append(
            "High token entropy: the model shows notable uncertainty about next-token choices."
        )
    elif entropy_mean > 3.0:
        notes.append(
            "Moderate token entropy: some uncertainty in generation."
        )

    # NLL thresholds
    if nll > 4.0:
        notes.append(
            "Very high NLL: the model assigned low probability to its own output tokens."
        )
    elif nll > 3.0:
        notes.append(
            "High NLL: the model was not highly confident in the tokens it chose."
        )
    elif nll > 2.0:
        notes.append(
            "Moderate NLL: confidence is acceptable but not high."
        )

    return notes or ["Uncertainty signals are within normal bounds."]


# ─────────────────────────────────────────────────────────────────────────────
# Clarification Questions
# ─────────────────────────────────────────────────────────────────────────────

def clarification_questions() -> List[str]:
    """
    Return clarification questions to show when HRI is high.

    These questions help users provide more specific context,
    which may reduce hallucination risk on a follow-up query.
    """
    return [
        "Can you specify the exact entity, date, or location the answer must be grounded on?",
        "Do you want a strictly factual answer (must be verifiable) or a conceptual explanation?",
        "What source or context should the response rely on (notes, paper, dataset, etc.)?",
    ]
