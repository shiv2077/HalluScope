"""Tests for the pure pieces of eval/grader_calibration: the threshold sweep
must pick the threshold that MAXIMISES agreement with human labels, and the
inter-annotator kappa must behave. Cosines are injected, so no SBERT load."""

from __future__ import annotations

from eval.grader_calibration import (
    sweep_agreement,
    best_threshold,
    inter_annotator_kappa,
    THRESHOLDS,
)


def _separable():
    # cosine high => human said 'correct'; cosine low => 'incorrect'
    cosines = [0.90, 0.80, 0.70, 0.20, 0.30, 0.40]
    labels = ["correct", "correct", "correct", "incorrect", "incorrect", "incorrect"]
    return cosines, labels


def test_sweep_covers_grid():
    cos, lab = _separable()
    rows = sweep_agreement(cos, lab, THRESHOLDS)
    assert rows[0]["threshold"] == 0.40
    assert rows[-1]["threshold"] == 0.80
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)


def test_best_threshold_maximises_human_agreement():
    cos, lab = _separable()
    rows = sweep_agreement(cos, lab, THRESHOLDS)
    best = best_threshold(rows)
    # cleanly separable in (0.40, 0.70] -> perfect agreement is achievable
    assert abs(best["accuracy"] - 1.0) < 1e-9
    assert abs(best["kappa"] - 1.0) < 1e-9
    assert 0.40 < best["threshold"] <= 0.70
    # it must be the global max, not a bimodal-looking point
    assert best["accuracy"] == max(r["accuracy"] for r in rows)


def test_extreme_thresholds_lose_agreement():
    cos, lab = _separable()
    rows = {round(r["threshold"], 2): r for r in sweep_agreement(cos, lab, THRESHOLDS)}
    assert rows[0.40]["accuracy"] < 1.0   # 0.40 lets a low-cosine 'incorrect' pass as correct
    assert rows[0.80]["accuracy"] < 1.0   # 0.80 marks a high-cosine 'correct' as incorrect


def test_inter_annotator_kappa_perfect_and_imperfect():
    a = ["correct", "incorrect", "correct", "incorrect"]
    assert abs(inter_annotator_kappa(a, a) - 1.0) < 1e-9
    b = ["correct", "incorrect", "incorrect", "incorrect"]  # one disagreement
    assert inter_annotator_kappa(a, b) < 1.0
