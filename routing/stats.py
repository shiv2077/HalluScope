"""Shared statistics: percentile bootstrap (2000 resamples, seed 42), Wilson intervals, AUROC, DeLong.

Every reported number goes through boot() or wilson(). A number without an interval is a bug.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import ROOT

sys.path.insert(0, str(ROOT))
from eval.stats import delong_test  # noqa: E402

N_BOOT = 2000
SEED = 42


def auc(y, s):
    y = np.asarray(y)
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def boot(stat, *arrays, n_boot: int = N_BOOT, seed: int = SEED) -> dict | None:
    """Percentile bootstrap over rows. Degenerate resamples (stat returns None or nan) are skipped."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    est = stat(*arrays)
    if est is None or not np.isfinite(est) or n == 0:
        return None
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        v = stat(*(a[idx] for a in arrays))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n": int(n), "n_boot_valid": len(vals)}


def wilson(k: int, n: int, z: float = 1.96) -> dict | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"est": p, "lo": c - h, "hi": c + h, "k": int(k), "n": int(n)}


def auc_ci(y, s) -> dict | None:
    d = boot(auc, y, s)
    if d:
        y = np.asarray(y)
        d.update(pos=int(y.sum()), neg=int(len(y) - y.sum()))
    return d


def delong(y, a, b) -> dict:
    """DeLong p-value for AUROC(a) vs AUROC(b) plus a paired bootstrap CI on the difference."""
    auc_a, auc_b, p = delong_test(np.asarray(y), np.asarray(a, float), np.asarray(b, float))
    diff = boot(lambda yy, aa, bb: None if len(np.unique(yy)) < 2 else auc(yy, aa) - auc(yy, bb), y, a, b)
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "p_delong": float(p), "diff": diff}


def hoeffding_ucb(risk_hat: float, n: int, delta: float) -> float:
    return risk_hat + math.sqrt(math.log(1 / delta) / (2 * n))


def f(d: dict | None, k: int = 3) -> str:
    if d is None:
        return "n/a"
    return f"{d['est']:.{k}f} [{d['lo']:.{k}f}, {d['hi']:.{k}f}]"


# Hanley-McNeil AUROC power (same formulas as eval/audit/p2_power.py).
def _hm_var(a, n_pos, n_neg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)


def auc_power(n_pos: int, n_neg: int, a1: float = 0.65, alpha: float = 0.05) -> float:
    from scipy import stats as sps

    za = sps.norm.ppf(1 - alpha / 2)
    return float(sps.norm.cdf((a1 - 0.5 - za * math.sqrt(_hm_var(0.5, n_pos, n_neg))) / math.sqrt(_hm_var(a1, n_pos, n_neg))))


def n_required(prevalence: float, target: float = 0.80) -> dict | None:
    for n in range(4, 200000):
        n_pos = round(prevalence * n)
        n_neg = n - n_pos
        if n_pos >= 1 and n_neg >= 1 and auc_power(n_pos, n_neg) >= target:
            return {"n": n, "pos": n_pos, "neg": n_neg}
    return None
