"""
Risk module for HalluScope.

This module contains risk assessment components including the
Hallucination Risk Index (HRI), threshold classification, and
intervention triggers.
"""

from .hri import (
    sigmoid,
    compute_hri_v1,
    explain_v1,
    clarification_questions,
)

__all__ = [
    "sigmoid",
    "compute_hri_v1",
    "explain_v1",
    "clarification_questions",
]
