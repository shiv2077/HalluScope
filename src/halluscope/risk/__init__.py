"""
Risk module for HalluScope.

TokenBurn-inspired hallucination risk assessment:
- HRI computation with multiple uncertainty signals
- Risk classification (LOW/MEDIUM/HIGH)
- Explanation generation
- Clarification question triggers
"""

from .hri import (
    sigmoid,
    compute_hri,
    classify_risk,
    compute_hri_with_label,
    explain,
    get_clarification_questions,
    clarification_questions_high,
    clarification_questions_medium,
    # Legacy
    compute_hri_v1,
    explain_v1,
    # Constants
    DEFAULT_WEIGHTS,
    THRESHOLD_LOW,
    THRESHOLD_HIGH,
)

__all__ = [
    "sigmoid",
    "compute_hri",
    "classify_risk",
    "compute_hri_with_label",
    "explain",
    "get_clarification_questions",
    "clarification_questions_high",
    "clarification_questions_medium",
    "compute_hri_v1",
    "explain_v1",
    "DEFAULT_WEIGHTS",
    "THRESHOLD_LOW",
    "THRESHOLD_HIGH",
]
