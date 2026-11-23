"""Calibration is the product. Every cutoff is computed on the calibration split and written to
results/calibration.json with the calibration file hash and n. No function here accepts the test split.

calibrate_budget: score cutoff that flags the top `budget` fraction of calibration rows.
calibrate_risk:   Learn-Then-Test with a Hoeffding bound, fixed-sequence testing over 200 thresholds from 10% coverage.
required_n:       calibration size at which a threshold at a given coverage would pass the test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import data, stats

BUDGETS = (0.10, 0.20, 0.30)
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
N_GRID = 200
COVERAGES = (0.5, 0.7, 0.9)
LTT_START_COVERAGE = 0.10   # routing's fixed sequence starts here (ADR-034); None keeps the original start


def _calibration_frame() -> pd.DataFrame:
    from .gates import scored_labelled

    m = scored_labelled("calibration")
    return m


def calibrate_budget(scores: np.ndarray, budget: float) -> float:
    """Cutoff such that rows with score >= cutoff are the top `budget` fraction (ties resolved upward)."""
    s = np.sort(scores[np.isfinite(scores)])[::-1]
    k = max(1, int(round(budget * len(s))))
    return float(s[k - 1])


def calibrate_risk(scores: np.ndarray, incorrect: np.ndarray, alpha: float, delta: float = DELTA,
                   n_grid: int = N_GRID, start: float | None = None) -> dict:
    """Most permissive threshold t (accept rows with score <= t) whose Hoeffding UCB on selective risk
    is at or below alpha. Thresholds are tested from most conservative to most permissive and testing
    stops at the first failure (fixed-sequence testing under monotone risk), so no union bound is needed.

    start=None begins at the lowest score, where one accepted row puts the bound above 1, so the sequence
    can fail at its first step (ADR-028). itemtriage's preregistered H3 call keeps that. A start coverage
    begins the same fixed sequence at that quantile instead. The grid does not depend on labels.
    """
    ok = np.isfinite(scores)
    s, y = scores[ok], incorrect[ok].astype(int)
    grid = np.quantile(s, np.linspace(0 if start is None else start, 1, n_grid))
    best = None
    for t in grid:
        acc = s <= t
        n = int(acc.sum())
        if n == 0:
            continue
        risk = float(y[acc].mean())
        ucb = stats.hoeffding_ucb(risk, n, delta)
        if ucb <= alpha:
            best = {"threshold": float(t), "coverage_cal": n / len(s), "risk_cal": risk, "ucb": ucb, "n_accepted": n}
        else:
            break
    if best is None:
        return {"threshold": None, "reason": f"no threshold has Hoeffding UCB <= {alpha} at delta={delta} "
                                             f"with n={len(s)}; see the required_n table"}
    return best


def required_n(alpha: float, delta: float, target_coverage: float, risk_hat: float) -> int | None:
    """Smallest n such that a threshold accepting target_coverage * n rows with empirical risk risk_hat
    has Hoeffding UCB <= alpha. None if risk_hat >= alpha (no n suffices)."""
    if risk_hat >= alpha:
        return None
    n_acc = math.log(1 / delta) / (2 * (alpha - risk_hat) ** 2)
    return int(math.ceil(n_acc / target_coverage))


def required_n_table() -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        for cov in COVERAGES:
            rows.append({"alpha": alpha, "delta": DELTA, "target_coverage": cov,
                         "n_if_risk_hat_0": required_n(alpha, DELTA, cov, 0.0),
                         "n_if_risk_hat_half_alpha": required_n(alpha, DELTA, cov, alpha / 2)})
    return rows


def run() -> dict:
    m = _calibration_frame()
    n_amb = int(m["ambiguous"].sum())
    m = m[~m["ambiguous"]]
    y = (m["label"] == "INCORRECT").to_numpy()
    p = data.RESULTS / "calibration.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    out = {"calibration_hash": data.file_hash(data.path("calibration", "scores")),
           "labels_hash": data.file_hash(data.path("calibration", "labels")),
           "n": int(len(m)), "n_ambiguous_excluded": n_amb, "positives": int(y.sum()),
           "budgets": {}, "risk": {}, "required_n": required_n_table(),
           "length_residual": cur.get("length_residual")}
    for sc in data.SCORERS:
        s = m[sc].astype(float).to_numpy()
        out["budgets"][sc] = {str(b): calibrate_budget(s, b) for b in BUDGETS}
        out["risk"][sc] = {str(a): calibrate_risk(s, y, a, start=LTT_START_COVERAGE) for a in ALPHAS}
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def load() -> dict:
    p = data.RESULTS / "calibration.json"
    if not p.exists():
        raise FileNotFoundError("run `python -m routing.cli calibrate` first")
    return json.loads(p.read_text())
