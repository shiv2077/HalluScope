"""Pure statistics for HalluScope evaluation.

  * bootstrap_auroc_ci      — percentile bootstrap 95% CI for AUROC.
  * delong_test             — fast DeLong test for the difference between two
                              correlated AUROCs (Sun & Xu, 2014).
  * expected_calibration_error / brier_score / reliability_curve.

These operate on real (y_true, y_score) arrays. They compute NO numbers on
their own — eval/analyze.py invokes them on the human's real result CSVs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


# ── Bootstrap CI ─────────────────────────────────────────────────────
def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for AUROC. Returns (lo, point, hi)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue  # resample degenerate (single-class) draw is undefined
        aucs.append(roc_auc_score(yt, y_score[idx]))
    if not aucs:
        return point, point, point
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, point, hi


# ── Fast DeLong ──────────────────────────────────────────────────────
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Fast DeLong (Sun & Xu 2014). Returns (aucs, covariance)."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(positive[r, :])
        ty[r, :] = _midrank(negative[r, :])
        tz[r, :] = _midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> Tuple[float, float, float]:
    """Compare two correlated AUROCs on the same labels.

    Returns (auc_a, auc_b, p_value) where p_value tests H0: AUC_a == AUC_b.
    """
    from scipy import stats

    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)  # positives (label 1) first
    label_1_count = int(np.sum(y_true == 1))
    preds = np.vstack((np.asarray(score_a), np.asarray(score_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        # No variance in the difference => identical ordering => no difference.
        p = 1.0 if abs(aucs[0] - aucs[1]) < 1e-12 else 0.0
        return float(aucs[0]), float(aucs[1]), float(p)
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


# ── Calibration ──────────────────────────────────────────────────────
def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, bin_accuracy, bin_confidence, bin_counts)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin is closed on the right so prob == 1.0 lands in it
        mask = (
            (y_prob >= lo) & (y_prob < hi)
            if b < n_bins - 1
            else (y_prob >= lo) & (y_prob <= hi)
        )
        counts[b] = int(np.sum(mask))
        if counts[b] > 0:
            acc[b] = float(np.mean(y_true[mask]))
            conf[b] = float(np.mean(y_prob[mask]))
    return centers, acc, conf, counts


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE = sum_b (n_b/N) * |acc_b - conf_b|."""
    _, acc, conf, counts = reliability_curve(y_true, y_prob, n_bins)
    n = int(np.sum(counts))
    if n == 0:
        return 0.0
    return float(np.sum(counts / n * np.abs(acc - conf)))
