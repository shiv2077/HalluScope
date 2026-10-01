"""Routing policies.

A budget policy fixes the flag rate and lets the score cutoff float: it takes the quantile on the
batch being routed. A risk policy fixes the score cutoff (from results/calibration.json) and lets
the flag rate float. The two are never mixed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def flag_by_budget(scores: pd.Series, budget: float) -> pd.Series:
    """Flag the top ceil(budget * n) scores of this batch. Ties broken by index (qid) ascending.

    Label-free and calibration-free by construction: the only inputs are the scores of the rows
    in the batch and the budget. No file is read. NaN scores are never flagged.
    """
    assert 0 < budget < 1, f"budget must be in (0, 1), got {budget}"
    n = len(scores)
    k = math.ceil(budget * n)
    s = scores.astype(float)
    idx = np.array(s.index, dtype=object)
    vals = s.to_numpy()
    finite = np.isfinite(vals)
    # stable sort by qid ascending, then stable sort by score descending; NaN sinks to the end
    order = np.lexsort((idx.astype(str), np.where(finite, -vals, np.inf)))
    flagged = np.zeros(n, dtype=bool)
    top = [i for i in order[:k] if finite[i]]
    flagged[top] = True
    return pd.Series(flagged, index=scores.index)


def flag_by_risk(scores: pd.Series, cutoff: float) -> pd.Series:
    """Flag rows whose score is at or above a calibrated cutoff. The flag rate floats."""
    return scores.astype(float) >= cutoff


def batch_cutoff(scores: pd.Series, budget: float) -> float:
    """The score value the budget policy used as its cutoff on this batch (lowest flagged score)."""
    fl = flag_by_budget(scores, budget)
    return float(scores.astype(float)[fl].min()) if fl.any() else float("nan")
