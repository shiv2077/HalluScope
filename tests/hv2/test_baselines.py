"""Plumbing for the Stage-3 LR baselines: feature dims, bootstrap CI, fit/eval."""

import math

import numpy as np

from src.hv2.baselines import ACCESS_CLASSES, bootstrap_auroc, extract_features, fit_eval
from src.hv2.hdf5_store import GenRecord


def _rec(T=6, K=100, L=25, H=64, P=4):
    rng = np.random.default_rng(1)
    return GenRecord(
        key="m/q/main",
        prompt="p",
        text="The capital is Paris and 1 more thing",
        token_ids=rng.integers(0, 1000, T),
        top_logprobs=-rng.random((T, K)).astype(np.float32),
        top_token_ids=rng.integers(0, 1000, (T, K)),
        functionals=rng.random((T, 3)).astype(np.float32),
        hidden_states=rng.standard_normal((P, L, H)).astype(np.float16),
        hidden_pos_index=np.arange(P),
        hidden_pos_kind=["last_prompt", "first_answer", "last_answer", "exact_answer"],
        answer_token_indices=np.array([3]),
    )


def test_extract_features_finite_per_class():
    rec = _rec(H=64)
    for ac in ACCESS_CLASSES:
        v = extract_features(rec, ac)
        assert v.ndim == 1 and np.isfinite(v).all()
    assert extract_features(rec, "hidden").shape[0] == 64  # H dims
    # cumulative levels grow
    assert len(extract_features(rec, "topk")) > len(extract_features(rec, "text"))
    assert len(extract_features(rec, "dist")) > len(extract_features(rec, "topk"))


def test_bootstrap_auroc_separable():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)])
    res = bootstrap_auroc(y, scores, n=300)
    assert res["auroc"] > 0.9
    assert res["ci_low"] <= res["auroc"] <= res["ci_high"]


def test_bootstrap_single_class_is_nan():
    res = bootstrap_auroc(np.zeros(10, dtype=int), np.random.rand(10))
    assert math.isnan(res["auroc"])


def test_fit_eval_recovers_signal():
    rng = np.random.default_rng(2)
    X_tr = np.vstack([rng.normal(0, 1, (40, 3)), rng.normal(2, 1, (40, 3))])
    y_tr = np.array([0] * 40 + [1] * 40)
    X_va = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(2, 1, (20, 3))])
    y_va = np.array([0] * 20 + [1] * 20)
    scores = fit_eval(X_tr, y_tr, X_va, y_va)
    assert bootstrap_auroc(y_va, scores, n=200)["auroc"] > 0.8
