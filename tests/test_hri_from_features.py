"""hri_from_features lets fit_weights / ablation / transfer recompute HRI from
logged feature columns with arbitrary weights. It must agree with compute_hri
on the same inputs and respond to weight changes."""

from __future__ import annotations

from backend.app.risk.hri_engine import compute_hri, hri_from_features
from backend.app import config_store


def test_matches_compute_hri_api_mode():
    logprobs = [-0.5, -0.6, -0.4, -0.55, -0.5, -0.6]
    res = compute_hri(logprobs)
    feats = {
        "logprob_variance": res["metrics"]["logprob_variance"],
        "confidence_decay": res["metrics"]["confidence_decay"],
        "abs_mean_logprob": abs(res["metrics"]["mean_logprob"]),
    }
    hri2 = hri_from_features(feats, mode="api")
    assert abs(hri2 - res["hri"]) < 1e-3


def test_responds_to_custom_weights():
    feats = {"logprob_variance": 0.5, "confidence_decay": 0.3, "abs_mean_logprob": 0.5}
    base = config_store.weights()
    hot = {
        "sigmoid": {"scale": 1.0, "bias": 0.0},
        "api": {
            "logprob_variance": 100.0,
            "confidence_decay": 0.0,
            "abs_mean_logprob": 0.0,
        },
    }
    cold = {
        "sigmoid": {"scale": 1.0, "bias": 0.0},
        "api": {
            "logprob_variance": -100.0,
            "confidence_decay": 0.0,
            "abs_mean_logprob": 0.0,
        },
    }
    assert hri_from_features(feats, mode="api", weights=hot) > 0.9
    assert hri_from_features(feats, mode="api", weights=cold) < 0.1
