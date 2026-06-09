"""
HalluScope – Intervention Service

Decides whether to intervene based on the HRI score.

The intervention trigger is the SAME threshold as the HIGH classification
band: ``config/thresholds.json -> risk_high`` (read via config_store). There
is exactly one source of truth — a response classified HIGH by
hri_engine.classify_risk is exactly the set of responses that intervene.

When intervention triggers, the original LLM answer is suppressed and
replaced with a deterministic clarification request.
"""

from backend.app import config_store


def should_intervene(hri: float) -> bool:
    """Return True when the HRI is at/above the HIGH threshold (intervene)."""
    return hri >= config_store.risk_high()


def decide_answer(
    model_text: str, hri: float, apply_intervention: bool, user_message: str
):
    """Resolve what text to display and whether the real answer was suppressed.

    Returns (displayed_text, intervention_fired, suppressed).

      * ``intervention_fired`` is the HRI decision (hri >= risk_high) and is
        ALWAYS reported, so downstream analysis can compute the
        false-suppression rate even in eval mode.
      * The real model answer is replaced by a clarification ONLY when the
        decision fires AND ``apply_intervention`` is True (live serving). In
        eval mode (apply_intervention=False) the real answer is preserved for
        grading — otherwise the benchmark would grade the clarification text.
    """
    fired = should_intervene(hri)
    if fired and apply_intervention:
        return generate_clarification(user_message), True, True
    return model_text, fired, False


def generate_clarification(user_message: str) -> str:
    """Produce a clarification message when risk is too high.

    Deterministic template — no LLM call needed.
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
