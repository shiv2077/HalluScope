"""
Hallucination Risk Index (HRI) — risk classification.

Only the sigmoid and classify_risk functions are used by V4.
The actual HRI computation happens in llm_backend_v4.py.
"""

from __future__ import annotations

import math

# Thresholds
THRESHOLD_LOW = 0.45
THRESHOLD_HIGH = 0.70


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def classify_risk(score: float) -> str:
    """Classify HRI score → LOW / MEDIUM / HIGH."""
    if score >= THRESHOLD_HIGH:
        return "HIGH"
    if score >= THRESHOLD_LOW:
        return "MEDIUM"
    return "LOW"
