"""Tests for the (pure) pieces of eval/fit_weights: feature-matrix assembly,
weight-dict construction, and that a fit on separable synthetic data yields
weights whose sign points the right way."""

from __future__ import annotations

from eval.fit_weights import build_xy, to_weight_block, fit_logistic


def _rows():
    # 'logprob_variance' perfectly separates: high var => mismatch(1).
    rows = []
    for v in [0.05, 0.1, 0.15, 0.2]:
        rows.append({"logprob_variance": str(v), "confidence_decay": "0.0",
                     "abs_mean_logprob": "0.5", "model_match": "match"})
    for v in [0.8, 0.85, 0.9, 0.95]:
        rows.append({"logprob_variance": str(v), "confidence_decay": "0.0",
                     "abs_mean_logprob": "0.5", "model_match": "mismatch"})
    return rows


FEATS = ["logprob_variance", "confidence_decay", "abs_mean_logprob"]


def test_build_xy_shapes_and_labels():
    X, y, used = build_xy(_rows(), FEATS)
    assert X.shape == (8, 3)
    assert list(y) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert used == 8


def test_build_xy_skips_rows_with_missing_features():
    rows = _rows() + [{"logprob_variance": "", "confidence_decay": "0.0",
                       "abs_mean_logprob": "0.5", "model_match": "mismatch"}]
    X, y, used = build_xy(rows, FEATS)
    assert used == 8  # the empty-feature row is skipped


def test_fit_recovers_positive_sign_for_separating_feature():
    X, y, _ = build_xy(_rows(), FEATS)
    coefs, intercept = fit_logistic(X, y, seed=0)
    block = to_weight_block(FEATS, coefs, intercept)
    # logprob_variance separates upward -> its weight must be the largest, > 0.
    assert block["logprob_variance"] > 0
    assert block["logprob_variance"] == max(block[f] for f in FEATS)
    assert block["sigmoid"]["scale"] == 1.0
    assert "source" not in block  # source tag is added by the file writer
