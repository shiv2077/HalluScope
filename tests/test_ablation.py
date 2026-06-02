"""Tests for the pure ablation helpers (leave-one-out reweighting and
single-feature AUROC)."""

from __future__ import annotations

import math

from eval.ablation import loo_weights, single_feature_auroc, auroc_from_hri


def test_loo_drops_key_and_renormalizes_to_preserve_mass():
    block = {"a": 0.4, "b": 0.3, "c": 0.3, "sigmoid": {"scale": 10.0, "bias": -3.0}}
    out = loo_weights(block, "a")
    assert "a" not in out
    # remaining mass preserved: original feature sum was 1.0
    assert math.isclose(out["b"] + out["c"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["b"], 0.5, abs_tol=1e-9)
    assert out["sigmoid"] == block["sigmoid"]  # sigmoid untouched


def test_single_feature_auroc_separating_feature_is_one():
    rows = []
    for v in [0.1, 0.2, 0.3]:
        rows.append({"logprob_variance": str(v), "model_match": "match"})
    for v in [0.7, 0.8, 0.9]:
        rows.append({"logprob_variance": str(v), "model_match": "mismatch"})
    auc = single_feature_auroc(rows, "logprob_variance")
    assert abs(auc - 1.0) < 1e-9


def test_auroc_from_hri_runs_over_feature_rows():
    rows = []
    for v in [0.05, 0.1, 0.15]:
        rows.append({"logprob_variance": str(v), "confidence_decay": "0.0",
                     "abs_mean_logprob": "0.5", "model_match": "match"})
    for v in [0.8, 0.9, 0.95]:
        rows.append({"logprob_variance": str(v), "confidence_decay": "0.0",
                     "abs_mean_logprob": "0.5", "model_match": "mismatch"})
    block = {"logprob_variance": 10.0, "confidence_decay": 0.0,
             "abs_mean_logprob": 0.0, "sigmoid": {"scale": 1.0, "bias": -4.0}}
    auc = auroc_from_hri(rows, ["logprob_variance", "confidence_decay",
                                "abs_mean_logprob"], block, mode="api")
    assert auc == 1.0  # variance perfectly separates
