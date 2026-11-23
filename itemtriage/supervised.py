"""The supervised secondary scorer `lr`: L2 logistic regression on standardized label-free features.

Fit on calibration fit only, with C chosen from {0.01, 0.1, 1, 10} by primary-class average precision on
validation. Test labels are never read here. Scores every item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import rank

C_GRID = (0.01, 0.1, 1.0, 10.0)


def fit_score(frame: pd.DataFrame, features: list[str], y_col: str, tie_col: str, extra: pd.DataFrame | None = None) -> dict:
    """frame: all items with features, `fold`, and y_col (NaN for unknown). Returns scores for every item."""
    X = frame[features].to_numpy(float)
    if extra is not None:
        X = np.hstack([X, extra.to_numpy(float)])
    fit = (frame["fold"] == "calibration_fit") & frame[y_col].notna()
    val = (frame["fold"] == "validation") & frame[y_col].notna()
    assert not (frame.loc[fit | val, "fold"] == "test").any()
    mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    yf = frame.loc[fit, y_col].to_numpy(int)
    yv = frame.loc[val, y_col].to_numpy(float)
    tv = frame.loc[val, tie_col].to_numpy(float)
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000).fit(Z[fit], yf)
        ap = rank.average_precision(yv, m.decision_function(Z[val]), tv)
        if best is None or ap > best[0] + 1e-12:
            best = (ap, C, m)
    ap, C, m = best
    return {"scores": m.decision_function(Z), "C": C, "validation_ap": ap,
            "coef": dict(zip(features + (list(extra.columns) if extra is not None else []), m.coef_[0].tolist())),
            "intercept": float(m.intercept_[0]), "n_fit": int(fit.sum())}


PCA_DIM = 64
OOF_FOLDS = 5


def fit_probe(X: np.ndarray, frame: pd.DataFrame, y_col: str, tie_col: str) -> dict:
    """H2 probe: PCA to 64 dimensions fit on calibration fit, then an L2 logistic regression with C chosen by
    validation average precision. Rows of calibration fit get out-of-fold scores (5 subject-grouped folds,
    PCA refit inside each fold) so a stacked model never sees a probe score that was fit on the same row's label.
    """
    from sklearn.decomposition import PCA

    fold = frame["fold"].to_numpy()
    y = frame[y_col].to_numpy(float)
    known = ~np.isnan(y)
    fit = (fold == "calibration_fit") & known
    val = (fold == "validation") & known
    tv = frame[tie_col].to_numpy(float)

    def train(rows, C):
        pca = PCA(n_components=PCA_DIM, random_state=42).fit(X[rows])
        m = LogisticRegression(C=C, max_iter=5000).fit(pca.transform(X[rows]), y[rows].astype(int))
        return lambda R: m.decision_function(pca.transform(X[R]))

    best = None
    for C in C_GRID:
        f_ = train(fit, C)
        ap = rank.average_precision(y[val], f_(val), tv[val])
        if best is None or ap > best[0] + 1e-12:
            best = (ap, C, f_)
    ap, C, full = best
    scores = full(np.ones(len(frame), bool))
    subj = frame["subject"].to_numpy()
    fit_subjects = sorted(set(subj[fit]))
    groups = {s: i % OOF_FOLDS for i, s in enumerate(fit_subjects)}
    for g in range(OOF_FOLDS):
        hold = np.array([fold[i] == "calibration_fit" and groups.get(subj[i]) == g for i in range(len(frame))])
        tr = fit & ~hold
        scores[hold] = train(tr, C)(hold)
    return {"scores": scores, "C": C, "validation_ap": ap, "pca_dim": PCA_DIM, "oof_folds": OOF_FOLDS}
