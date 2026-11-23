"""Tie handling is permutation-invariant; the 4PL forced-ceiling contrast ranks planted mislabels first."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from itemtriage import irt, rank  # noqa: E402


def test_tied_block_uses_expectation_and_is_permutation_invariant():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s1 = np.array([5, 5, 5, 5, 1, 1, 1, 0, 0, 0], float)
    s2 = np.zeros(10)
    # k = 2 cuts through the block of four rows at score 5 holding 2 positives: expected 2 * 2/4 = 1
    assert rank.expected_flagged(y, s1, s2, 2) == (1.0, 4)
    rng = np.random.RandomState(0)
    for _ in range(20):
        p = rng.permutation(10)
        assert rank.expected_flagged(y[p], s1[p], s2[p], 2) == (1.0, 4)
        assert rank.average_precision(y[p], s1[p], s2[p]) == rank.average_precision(y, s1, s2)


def test_secondary_key_breaks_ties():
    y = np.array([0, 1, 0, 0])
    s1 = np.array([1, 1, 0, 0], float)
    s2 = np.array([0.1, 0.9, 0, 0])
    assert rank.expected_flagged(y, s1, s2, 1) == (1.0, 0)
    assert rank.precision_at(y, s1, s2, 0.25) == 1.0


def simulate(n_models, n_items=400, n_bad=40, seed=0):
    """Responses from a 3PL world. Planted mislabels: the key is wrong, so stronger models answer it less."""
    rng = np.random.RandomState(seed)
    theta = np.sort(rng.normal(0, 1, n_models))
    b = rng.uniform(-1.5, 1.5, n_items)
    a = rng.uniform(0.8, 2.0, n_items)
    p = 0.25 + 0.75 / (1 + np.exp(-a[:, None] * (theta[None, :] - b[:, None])))
    bad = np.zeros(n_items, bool)
    bad[rng.choice(n_items, n_bad, replace=False)] = True
    p[bad] = 0.25 * (1 - 1 / (1 + np.exp(-1.5 * theta[None, :])))  # strong models avoid the wrong key
    Y = (rng.uniform(size=p.shape) < p).astype(float)
    return Y, bad


def _auroc(score, bad):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(bad, score)


def test_forced_ceiling_ranks_planted_mislabels_with_a_large_panel():
    Y, bad = simulate(60)
    groups = np.zeros(len(Y), int)
    s1 = irt.fit_stage1(Y, groups, 0.25)
    assert s1["n_kept"] > 100
    d_full = irt.delta_full(Y, s1["theta"], 0.25)
    d_con = irt.delta_constrained(Y, s1["theta"], groups, s1["a_group"], 0.25)
    assert _auroc(d_full, bad) > 0.9
    assert _auroc(d_con, bad) > 0.9


def test_small_panel_still_carries_signal():
    Y, bad = simulate(7, seed=1)
    groups = np.zeros(len(Y), int)
    s1 = irt.fit_stage1(Y, groups, 0.25)
    d_con = irt.delta_constrained(Y, s1["theta"], groups, s1["a_group"], 0.25)
    assert _auroc(d_con, bad) > 0.7
