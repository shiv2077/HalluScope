"""Cross-model transfer: rescore a feature CSV's rows under foreign weights."""

from __future__ import annotations

from eval.score_features import rescore_rows


def test_rescore_applies_given_weights():
    rows = [
        {"logprob_variance": "0.1", "confidence_decay": "0.0", "abs_mean_logprob": "0.5"},
        {"logprob_variance": "0.9", "confidence_decay": "0.0", "abs_mean_logprob": "0.5"},
    ]
    weights = {"sigmoid": {"scale": 1.0, "bias": -4.0},
               "api": {"logprob_variance": 10.0, "confidence_decay": 0.0,
                       "abs_mean_logprob": 0.0}}
    out = rescore_rows(rows, mode="api", weights=weights)
    # higher variance -> higher rescored HRI
    assert out[1]["hri"] > out[0]["hri"]
    assert all(0.0 <= r["hri"] <= 1.0 for r in out)
