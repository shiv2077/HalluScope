"""Budget metrics that never depend on row order. Label-free ranking; labels enter only as y.

Rows are ordered by (score, declared tie-break key), both descending. Rows still tied on both keys form a
block; when a block straddles the budget cutoff, the flagged positives count the block's expected share
under uniform resolution: needed x block positives / block size. Results are invariant to row permutation.
"""

from __future__ import annotations

import math

import numpy as np


def expected_flagged(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, k: int) -> tuple[float, int]:
    """Expected positives among the top k, and the size of the tied block at the cutoff (0 if none straddles)."""
    y, s1, s2 = (np.asarray(v, float) for v in (y, s1, s2))
    order = np.lexsort((-s2, -s1))
    ys, a, b = y[order], s1[order], s2[order]
    if k <= 0:
        return 0.0, 0
    if k >= len(ys):
        return float(ys.sum()), 0
    ca, cb = a[k - 1], b[k - 1]
    block = (a == ca) & (b == cb)
    first = int(np.argmax(block))
    size = int(block.sum())
    above = ys[:first].sum()
    need = k - first
    if need >= size:
        return float(above + ys[block].sum()), 0
    return float(above + need * ys[block].sum() / size), size


def budget_k(n: int, budget: float) -> int:
    assert 0 < budget < 1
    return math.ceil(budget * n)


def precision_at(y, s1, s2, budget: float) -> float:
    k = budget_k(len(y), budget)
    return expected_flagged(y, s1, s2, k)[0] / k


def recall_at(y, s1, s2, budget: float) -> float | None:
    tot = float(np.sum(y))
    if tot == 0:
        return None
    return expected_flagged(y, s1, s2, budget_k(len(y), budget))[0] / tot


def straddle_size(y, s1, s2, budget: float) -> int:
    return expected_flagged(y, s1, s2, budget_k(len(y), budget))[1]


def average_precision(y, s1, s2) -> float | None:
    """Exact expected AP when ties are resolved uniformly at random, by linearity over positives.

    In a tied block of size m with p positives, after f rows holding A positives, a positive lands at block
    position t with probability 1/m and then has A + 1 + (t - 1)(p - 1)/(m - 1) expected positives at or
    above it. Its expected precision is that over (f + t). AP sums these over positives and divides by P.
    """
    y, s1, s2 = (np.asarray(v, float) for v in (y, s1, s2))
    P = y.sum()
    if P == 0:
        return None
    order = np.lexsort((-s2, -s1))
    ys, a, b = y[order], s1[order], s2[order]
    n = len(ys)
    starts = np.flatnonzero(np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])])
    ends = np.r_[starts[1:], n]
    total, above = 0.0, 0.0
    for f, e in zip(starts, ends):
        m = e - f
        p = ys[f:e].sum()
        if p:
            t = np.arange(1, m + 1)
            inc = (t - 1) * (p - 1) / (m - 1) if m > 1 else np.zeros(1)
            total += p / m * np.sum((above + 1 + inc) / (f + t))
        above += p
    return float(total / P)
