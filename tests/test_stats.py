"""Tests for the pure statistics behind eval/analyze.py: bootstrap AUROC CI,
DeLong test, ECE, Brier, reliability curve."""

from __future__ import annotations

import numpy as np

from eval.stats import (
    bootstrap_auroc_ci,
    delong_test,
    expected_calibration_error,
    brier_score,
    reliability_curve,
)


def test_brier_matches_definition():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    assert abs(brier_score(y, p) - np.mean((p - y) ** 2)) < 1e-12


def test_ece_zero_for_perfectly_calibrated_hard_predictions():
    # Probabilities equal to the true labels => perfectly calibrated.
    y = np.array([0, 0, 1, 1, 1, 0])
    p = y.astype(float)
    assert expected_calibration_error(y, p, n_bins=10) < 1e-9


def test_ece_positive_for_overconfident_predictions():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99])  # always confident, 50% right
    assert expected_calibration_error(y, p, n_bins=10) > 0.4


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.RandomState(0)
    y = np.array([0] * 50 + [1] * 50)
    # scores correlated with label but noisy
    s = np.concatenate([rng.rand(50) * 0.6, 0.4 + rng.rand(50) * 0.6])
    lo, point, hi = bootstrap_auroc_ci(y, s, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_delong_identical_scores_no_difference():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7, 0.85, 0.25])
    auc_a, auc_b, p = delong_test(y, s, s)
    assert abs(auc_a - auc_b) < 1e-9
    assert p > 0.99  # no detectable difference
    assert 0.0 <= p <= 1.0


def test_delong_returns_valid_pvalue_for_different_scorers():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    good = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7, 0.85, 0.25])
    bad = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.3, 0.15, 0.75])
    auc_a, auc_b, p = delong_test(y, good, bad)
    assert auc_a > auc_b
    assert 0.0 <= p <= 1.0


def test_reliability_curve_shapes():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
    centers, acc, conf, counts = reliability_curve(y, p, n_bins=5)
    assert len(centers) == len(acc) == len(conf) == len(counts) == 5
    assert int(np.sum(counts)) == len(y)
