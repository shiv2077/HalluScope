"""Direction tests for Stage 2 internal signals: Logit Lens Divergence (LLD)
and Confidence-Stability Gap (CSG). Pure functions, synthetic inputs — no
model load required."""

from __future__ import annotations

import numpy as np

from backend.app.metrics.internal_metrics import (
    logit_lens_divergence_token,
    confidence_stability_gap,
    semantic_confidence,
    mean_logit_lens_divergence,
    mean_confidence_stability_gap,
    aggregate_internal_signals,
)


# ── LLD ──────────────────────────────────────────────────────────────
def _identity_unembed(d: int = 3) -> np.ndarray:
    return np.eye(d, dtype=np.float32)


def test_lld_zero_when_layers_agree():
    h = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    layers = [h.copy(), h.copy(), h.copy()]
    lld = logit_lens_divergence_token(layers, _identity_unembed())
    assert lld < 1e-6


def test_lld_positive_when_layers_disagree():
    layers = [
        np.array([6.0, 0.0, 0.0], dtype=np.float32),  # predicts token 0
        np.array([0.0, 6.0, 0.0], dtype=np.float32),  # predicts token 1
        np.array([0.0, 0.0, 6.0], dtype=np.float32),  # predicts token 2
    ]
    lld = logit_lens_divergence_token(layers, _identity_unembed())
    assert lld > 1.0


def test_lld_disagreement_exceeds_agreement():
    agree = logit_lens_divergence_token(
        [np.array([5.0, 0, 0], np.float32)] * 3, _identity_unembed())
    disagree = logit_lens_divergence_token(
        [np.array([5.0, 0, 0], np.float32), np.array([0, 5.0, 0], np.float32)],
        _identity_unembed())
    assert disagree > agree


# ── CSG ──────────────────────────────────────────────────────────────
def test_semantic_confidence_maps_cosine_to_unit_interval():
    ctx = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(semantic_confidence(ctx, ctx) - 1.0) < 1e-6          # cos=1 -> 1
    assert abs(semantic_confidence(ctx, -ctx) - 0.0) < 1e-6         # cos=-1 -> 0
    orth = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(semantic_confidence(ctx, orth) - 0.5) < 1e-6         # cos=0 -> 0.5


def test_csg_small_when_token_and_semantic_confidence_agree():
    ctx = np.array([1.0, 0.0], dtype=np.float32)
    gap = confidence_stability_gap(token_top1_prob=1.0, token_hidden=ctx,
                                   context_vector=ctx)
    assert gap < 1e-6  # c_token=1, c_sem=1 -> gap 0


def test_csg_large_when_confident_but_semantically_incoherent():
    ctx = np.array([1.0, 0.0], dtype=np.float32)
    aligned = confidence_stability_gap(0.9, ctx, ctx)
    incoherent = confidence_stability_gap(0.9, -ctx, ctx)  # high prob, cos=-1
    assert incoherent > aligned
    assert 0.0 <= incoherent <= 1.0


# ── sequence aggregators ─────────────────────────────────────────────
def test_mean_lld_over_tokens():
    unembed = _identity_unembed()
    agree = [np.array([5.0, 0, 0], np.float32)] * 3
    disagree = [np.array([5.0, 0, 0], np.float32),
                np.array([0, 5.0, 0], np.float32),
                np.array([0, 0, 5.0], np.float32)]
    # two tokens: one stable, one churning
    m = mean_logit_lens_divergence([agree, disagree], unembed)
    m_stable = mean_logit_lens_divergence([agree, agree], unembed)
    assert m > m_stable


def test_mean_csg_rises_with_incoherent_token():
    a = np.array([1.0, 0.0], np.float32)
    coherent = mean_confidence_stability_gap([1.0, 1.0], [a, a])
    # second token confident (prob 1.0) but anti-aligned with the context mean
    incoherent = mean_confidence_stability_gap([1.0, 1.0], [a, -a])
    assert incoherent > coherent


def test_aggregate_returns_none_lld_csg_in_api_mode():
    agg = aggregate_internal_signals(None, None)
    assert agg["mean_lld"] is None
    assert agg["mean_csg"] is None
