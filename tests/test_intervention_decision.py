"""The benchmark must capture the model's REAL answer for grading, even when
HRI is high enough to intervene. decide_answer separates the intervention
DECISION (always reported) from whether the answer is actually SUPPRESSED
(only when apply_intervention is True, i.e. live serving — not eval)."""

from __future__ import annotations

from backend.app import config_store
from backend.app.services.intervention_service import decide_answer

HIGH = config_store.risk_high() + 0.05
LOW = config_store.risk_low() - 0.01


def test_live_serving_suppresses_high_risk_answer():
    displayed, fired, suppressed = decide_answer(
        "real model answer", HIGH, apply_intervention=True, user_message="q")
    assert fired is True and suppressed is True
    assert displayed != "real model answer"  # clarification shown to user


def test_eval_mode_keeps_real_answer_but_still_reports_decision():
    displayed, fired, suppressed = decide_answer(
        "real model answer", HIGH, apply_intervention=False, user_message="q")
    assert fired is True          # decision still recorded (for suppression-rate)
    assert suppressed is False    # but the real answer is NOT replaced
    assert displayed == "real model answer"


def test_low_risk_never_intervenes():
    displayed, fired, suppressed = decide_answer(
        "real model answer", LOW, apply_intervention=True, user_message="q")
    assert fired is False and suppressed is False
    assert displayed == "real model answer"
