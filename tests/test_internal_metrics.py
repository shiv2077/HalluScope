"""Direction + scaling tests for local-mode internal signals.

The pre-fix `semantic_energy` returned the raw free energy E = -logsumexp(z)
and the HRI engine consumed it via abs(E)/35, which (a) is not bounded to
[0, 1], (b) is dominated by logit *magnitude* rather than distribution
*shape*, and (c) assigned HIGHER risk to MORE confident (peaked) tokens.

These tests fix the contract: semantic energy is a bounded flatness signal
where FLATTER (more uncertain) distributions score HIGHER.
"""

from __future__ import annotations

import math

import numpy as np

from backend.app.metrics.internal_metrics import (
    semantic_energy,
    representation_collapse,
)


def _peaked(vocab: int = 1000) -> np.ndarray:
    z = np.zeros(vocab, dtype=np.float32)
    z[0] = 20.0  # one token dominates -> near-deterministic next token
    return z


def _uniform(vocab: int = 1000) -> np.ndarray:
    return np.zeros(vocab, dtype=np.float32)  # all logits equal -> maximally flat


# ── Semantic energy: direction ───────────────────────────────────────
def test_semantic_energy_flatter_is_higher():
    e_peaked, e_flat = semantic_energy([_peaked(), _uniform()])
    assert e_flat > e_peaked, (
        f"flat distribution must score higher energy than peaked "
        f"(flat={e_flat}, peaked={e_peaked})"
    )


# ── Semantic energy: bounded to [0, 1] ───────────────────────────────
def test_semantic_energy_is_bounded_unit_interval():
    energies = semantic_energy([_peaked(), _uniform(), _peaked(50), _uniform(50)])
    for e in energies:
        assert 0.0 <= e <= 1.0, f"energy {e} outside [0, 1]"


def test_semantic_energy_uniform_is_one_peaked_is_near_zero():
    e_peaked, e_uniform = semantic_energy([_peaked(), _uniform()])
    assert e_uniform == 1.0, f"uniform logits must be max flatness 1.0, got {e_uniform}"
    assert e_peaked < 0.05, f"strongly peaked logits must be near 0, got {e_peaked}"


# ── Semantic energy: shift-invariance (the magnitude-domination bug) ──
def test_semantic_energy_is_shift_invariant():
    """Adding a constant to every logit changes softmax by nothing, so the
    flatness energy must not move. The old -logsumexp formulation failed this."""
    base = _peaked()
    shifted = base + 100.0
    (e_base,) = semantic_energy([base])
    (e_shifted,) = semantic_energy([shifted])
    assert math.isclose(e_base, e_shifted, abs_tol=1e-6), (
        f"energy must be invariant to a global logit shift "
        f"(base={e_base}, shifted={e_shifted})"
    )


# ── Representation collapse: direction (unchanged behaviour, pinned) ──
def test_rcd_identical_states_collapse_to_zero():
    h = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scores = representation_collapse([h, h.copy()])
    assert len(scores) == 1
    assert math.isclose(scores[0], 0.0, abs_tol=1e-6)


def test_rcd_orthogonal_states_are_one():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    (score,) = representation_collapse([a, b])
    assert math.isclose(score, 1.0, abs_tol=1e-6)
