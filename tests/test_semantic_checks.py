"""Direction tests for Stage 3 post-generation semantic checks.

semantic_novelty takes an injectable ``embed_fn`` so these run without loading
SBERT; entity_expansion and claim_density use the dependency-free defaults.
"""

from __future__ import annotations

import numpy as np

from backend.app.metrics.semantic_checks import (
    semantic_novelty,
    entity_expansion,
    claim_density,
    compute_semantic_checks,
)


# Fake embedder: maps known strings to fixed unit vectors so cosine is exact.
def _fake_embed(texts):
    table = {
        "same": np.array([1.0, 0.0, 0.0]),
        "orthogonal": np.array([0.0, 1.0, 0.0]),
    }
    return np.stack([table.get(t.strip(), np.array([1.0, 0.0, 0.0])) for t in texts])


# ── semantic novelty ─────────────────────────────────────────────────
def test_novelty_zero_when_response_matches_prompt():
    nov = semantic_novelty("same", "same", embed_fn=_fake_embed)
    assert abs(nov - 0.0) < 1e-6


def test_novelty_high_when_response_unrelated():
    nov = semantic_novelty("same", "orthogonal", embed_fn=_fake_embed)
    assert nov > 0.9
    assert 0.0 <= nov <= 1.0


# ── entity expansion ─────────────────────────────────────────────────
def test_entity_expansion_counts_new_entities():
    # response introduces Brzezinski and Warsaw, not present in the prompt.
    exp = entity_expansion(
        prompt="Tell me about Poland.",
        response="Zbigniew Brzezinski was born in Warsaw, Poland.",
    )
    assert exp > 0.0
    assert 0.0 <= exp <= 1.0


def test_entity_expansion_zero_when_no_new_entities():
    exp = entity_expansion(
        prompt="Tell me about Warsaw.",
        response="Warsaw is nice.",
    )
    assert exp == 0.0


# ── claim density ────────────────────────────────────────────────────
def test_claim_density_higher_for_fact_dense_text():
    dense = claim_density(
        "Paris is the capital of France. It has 2 million people. "
        "The Seine flows through it."
    )
    sparse = claim_density(
        "It is honestly quite hard to say what might possibly be true here."
    )
    assert dense > sparse
    assert 0.0 <= dense <= 1.0


# ── aggregate ────────────────────────────────────────────────────────
def test_compute_semantic_checks_keys_and_bounds():
    out = compute_semantic_checks(
        "Tell me about Poland.",
        "Zbigniew Brzezinski was born in Warsaw in 1928.",
        embed_fn=_fake_embed,
    )
    for k in ("semantic_novelty", "entity_expansion", "claim_density"):
        assert k in out
        assert 0.0 <= out[k] <= 1.0
