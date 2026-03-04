"""
HalluScope v2 – Intervention Service

Decides whether to intervene based on the HRI score.

Risk thresholds:
    LOW     < 0.35   → respond normally
    MEDIUM  0.35–0.45 → respond with warning
    HIGH    >= 0.46   → block answer, ask clarification

When intervention triggers, the original LLM answer is suppressed
and replaced with a clarification request.
"""

INTERVENTION_THRESHOLD = 0.46  # matches RISK_HIGH_THRESHOLD


def should_intervene(hri: float) -> bool:
    """Return True when the HRI is high enough to warrant blocking."""
    return hri >= INTERVENTION_THRESHOLD


def generate_clarification(user_message: str) -> str:
    """Produce a clarification message when risk is too high.

    This is a deterministic template — no LLM call needed.
    """
    return (
        "⚠️ **High hallucination risk detected.**\n\n"
        "I'm not confident I can answer this accurately. "
        "The information requested may be incorrect or unverifiable.\n\n"
        "Could you help me by clarifying:\n"
        "• Are the names, dates, or titles in your question accurate?\n"
        "• Can you provide additional context or sources?\n"
        "• Would you like me to rephrase or narrow the question?"
    )
