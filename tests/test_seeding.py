"""Determinism guarantees for the shared seed utility."""

from __future__ import annotations

import random

import numpy as np

from backend.app.seeding import set_global_seed


def test_same_seed_reproduces_python_and_numpy_draws():
    set_global_seed(123)
    a = (random.random(), np.random.rand(3).tolist())
    set_global_seed(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_different_seeds_differ():
    set_global_seed(1)
    a = np.random.rand(5).tolist()
    set_global_seed(2)
    b = np.random.rand(5).tolist()
    assert a != b


def test_same_seed_reproduces_torch_draws():
    import torch

    set_global_seed(7)
    a = torch.rand(4)
    set_global_seed(7)
    b = torch.rand(4)
    assert torch.equal(a, b)
