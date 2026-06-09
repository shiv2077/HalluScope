"""Stage-3 PREVIEW detectors: one logistic regression per access class.

This is a preview of E2, not E2 itself. For each nested observation level
(text ⊆ topk ⊆ dist ⊆ hidden) we extract features from a cached GenRecord, fit
LR on the calib split, and score val, reporting AUROC with a 1000-resample
bootstrap CI — overall and per stratum (the KH/UH contrast is the whole point).

Feature levels are CUMULATIVE for a/b/c (each adds to the previous); the hidden
level uses the mean-pooled last-layer representation at the captured positions.
NEVER touches full-vocab logits (only the cached top-100 + functionals).
"""

from __future__ import annotations

import numpy as np

ACCESS_CLASSES = ("text", "topk", "dist", "hidden")
_HEDGE = ("i don't", "i do not", "not sure", "cannot", "can't", "unknown", "no idea", "unsure")


def _text_feats(rec) -> np.ndarray:
    t = rec.text.lower()
    return np.array(
        [
            len(rec.text),
            len(rec.token_ids),
            sum(c.isdigit() for c in rec.text),
            float(any(h in t for h in _HEDGE)),
            len(rec.text.split()),
        ],
        dtype=np.float32,
    )


def _topk_feats(rec) -> np.ndarray:
    lp = rec.top_logprobs.astype(np.float32)  # [T,K]
    top1 = lp[:, 0]
    gap = lp[:, 0] - lp[:, 1]
    mass = np.log(np.exp(lp).sum(axis=1) + 1e-9)  # log top-k probability mass
    return np.array(
        [top1.mean(), top1.min(), gap.mean(), np.exp(top1).mean(), mass.mean()], dtype=np.float32
    )


def _dist_feats(rec) -> np.ndarray:
    f = rec.functionals.astype(np.float32)  # [T,3] entropy,energy,margin
    return np.concatenate([f.mean(0), f.max(0), f[-1]]).astype(np.float32)  # 9 dims


def _hidden_feats(rec) -> np.ndarray:
    h = rec.hidden_states.astype(np.float32)  # [P,L,H]
    return h[:, -1, :].mean(axis=0)  # mean over captured positions, last layer -> [H]


def extract_features(rec, access_class: str) -> np.ndarray:
    if access_class == "text":
        return _text_feats(rec)
    if access_class == "topk":
        return np.concatenate([_text_feats(rec), _topk_feats(rec)])
    if access_class == "dist":
        return np.concatenate([_text_feats(rec), _topk_feats(rec), _dist_feats(rec)])
    if access_class == "hidden":
        return _hidden_feats(rec)
    raise ValueError(access_class)


def bootstrap_auroc(y: np.ndarray, scores: np.ndarray, n: int = 1000, seed: int = 42) -> dict:
    """AUROC + 95% bootstrap CI (1000 resamples). Returns NaNs if one class absent."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    scores = np.asarray(scores)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": int(len(y))}
    point = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(y))
    for _ in range(n):
        bi = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[bi])) < 2:
            continue
        boots.append(roc_auc_score(y[bi], scores[bi]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"auroc": point, "ci_low": float(lo), "ci_high": float(hi), "n": int(len(y))}


def fit_eval(X_tr, y_tr, X_va, y_va) -> np.ndarray:
    """Standardize + LR on train; return predicted hallucination probabilities on val."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_va)[:, 1]
