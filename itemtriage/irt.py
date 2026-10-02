"""Small-panel 4PL IRT with the forced-ceiling mislabel contrast of Land and Bikel (arXiv 2605.30504). Label-free.

Model: P(y_im = 1) = c_i + (d_i - c_i) * sigmoid(a_i (theta_m - b_i)), y = 1 when model m's answer is the key.
Stage 1, on items with item-total r > 0.2: constrained 3PL with one discrimination per subject, c = chance,
d = 1, b in [-2, 2], a in [1/15, 15], plus the anchor penalty lambda (mean(theta)^2 + (sd(theta) - 1)^2),
lambda = 10, fit by L-BFGS-B with analytic gradients. Stage 2, theta fixed: per item, refit (a, b, c) under
d = 0 and d = 1 and take delta = l(d=0) - l(d=1), with l the mean per-respondent log-likelihood.
The constrained variant fixes c = chance and a = the subject value, and refits b only.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

A_LO, A_HI = 1 / 15, 15.0
B_LO, B_HI = -2.0, 2.0
LAMBDA = 10.0
R_MIN = 0.2
EPS = 1e-6


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def item_total_r(Y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Pearson r between each item's correctness over models and the models' mean accuracy on the other items
    of the same group. NaN where either vector is constant."""
    r = np.full(Y.shape[0], np.nan)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        S = Y[idx].sum(axis=0)
        for i in idx:
            rest = (S - Y[i]) / max(len(idx) - 1, 1)
            y = Y[i]
            if y.std() > 0 and rest.std() > 0:
                r[i] = float(np.corrcoef(y, rest)[0, 1])
    return r


def fit_stage1(Y: np.ndarray, groups: np.ndarray, chance: float) -> dict:
    """Y: items x models binary. Returns theta per model, a per group, b per kept item, and the kept mask."""
    r = item_total_r(Y, groups)
    keep = np.nan_to_num(r, nan=-1) > R_MIN
    Yk, gk = Y[keep], groups[keep]
    ug, gi = np.unique(gk, return_inverse=True)
    n_i, n_m = Yk.shape
    n_g = len(ug)
    x0 = np.concatenate([np.zeros(n_m), np.ones(n_g), np.zeros(n_i)])

    def unpack(x):
        return x[:n_m], x[n_m:n_m + n_g], x[n_m + n_g:]

    def f(x):
        th, a, b = unpack(x)
        A = a[gi][:, None]
        z = A * (th[None, :] - b[:, None])
        s = _sig(z)
        p = np.clip(chance + (1 - chance) * s, EPS, 1 - EPS)
        ll = Yk * np.log(p) + (1 - Yk) * np.log(1 - p)
        N = ll.size
        mu, sd = th.mean(), th.std()
        loss = -ll.sum() / N + LAMBDA * (mu ** 2 + (sd - 1) ** 2)
        dz = ((Yk - p) / (p * (1 - p))) * (1 - chance) * s * (1 - s) / N  # d(ll/N)/dz
        g_th = -(dz * A).sum(axis=0)
        g_a_item = -(dz * (th[None, :] - b[:, None])).sum(axis=1)
        g_a = np.bincount(gi, weights=g_a_item, minlength=n_g)
        g_b = (dz * A).sum(axis=1)
        dmu = 2 * mu / n_m
        dsd = (2 * (sd - 1) * (th - mu) / (n_m * sd)) if sd > 0 else np.zeros(n_m)
        g_th = g_th + LAMBDA * (dmu + dsd)
        return loss, np.concatenate([g_th, g_a, g_b])

    bounds = [(None, None)] * n_m + [(A_LO, A_HI)] * n_g + [(B_LO, B_HI)] * n_i
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
    th, a, b = unpack(res.x)
    return {"theta": th, "a_group": dict(zip(ug.tolist(), a.tolist())), "kept": keep, "n_kept": int(keep.sum()),
            "converged": bool(res.success), "message": str(res.message), "loss": float(res.fun)}


def _item_ll(y, th, a, b, c, d):
    p = np.clip(c + (d - c) * _sig(a * (th - b)), EPS, 1 - EPS)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _best_ll(y, th, d, a0, starts):
    best = -np.inf
    for b0, c0 in starts:
        res = minimize(lambda v: -_item_ll(y, th, v[0], v[1], v[2], d), x0=[a0, b0, c0], method="L-BFGS-B",
                       bounds=[(A_LO, A_HI), (B_LO, B_HI), (0.0, 1.0)])
        best = max(best, -float(res.fun))
    return best


def delta_full(Y: np.ndarray, theta: np.ndarray, chance: float) -> np.ndarray:
    """Per-item forced-ceiling contrast with a, b, c free under d = 0 and d = 1. Multiple starts per fit."""
    starts = [(0.0, chance), (-1.5, 0.5), (1.5, 0.05)]
    return np.array([_best_ll(y, theta, 0.0, 1.0, starts) - _best_ll(y, theta, 1.0, 1.0, starts) for y in Y])


def _ll_b_only(y, th, a, c, d):
    best = -np.inf
    for b0 in (-1.5, 0.0, 1.5):
        res = minimize(lambda v: -_item_ll(y, th, a, v[0], c, d), x0=[b0], method="L-BFGS-B", bounds=[(B_LO, B_HI)])
        best = max(best, -float(res.fun))
    return best


def delta_constrained(Y: np.ndarray, theta: np.ndarray, groups: np.ndarray, a_group: dict, chance: float) -> np.ndarray:
    """Forced-ceiling contrast with c = chance, a = the subject's stage-1 value (1 if the subject had no kept item)."""
    out = np.empty(Y.shape[0])
    for i, (y, g) in enumerate(zip(Y, groups)):
        a = a_group.get(g, 1.0)
        out[i] = _ll_b_only(y, theta, a, chance, 0.0) - _ll_b_only(y, theta, a, chance, 1.0)
    return out
