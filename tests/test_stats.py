"""Tests for the DeLong test in routing.stats."""

from __future__ import annotations

import numpy as np

from routing.stats import delong_test

Y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
GOOD = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7, 0.85, 0.25])


def test_delong_identical_scores_no_difference():
    auc_a, auc_b, p = delong_test(Y, GOOD, GOOD)
    assert abs(auc_a - auc_b) < 1e-9
    assert p > 0.99


def test_delong_perfect_and_reversed_scorers():
    auc_a, auc_b, p = delong_test(Y, GOOD, -GOOD)
    assert auc_a == 1.0 and auc_b == 0.0
    assert 0.0 <= p <= 1.0
