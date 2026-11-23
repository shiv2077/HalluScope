"""Exploratory, post-unblinding: does the panel find bad items or only bad subjects?

The test fold was unblinded on 2 October 2026, so nothing here has a decision rule. Two label-free scorers:
subject_mean (each item gets the mean conf_w of its subject) and within_subject (conf_w minus subject_mean).
subject_mean has no secondary key, so ties inside a subject resolve in expectation and no item-level signal
leaks in at the cutoff. Labels join only in evaluate(). Writes results/itemtriage/explore.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from routing import stats

from . import evaluate, paths, rank, scorers

OUT = paths.RESULTS / "explore.json"
NAMES = ("conf_w", "subject_mean", "within_subject", "disagree", "length")
PAIRS = [("conf_w", "subject_mean"), ("within_subject", "length")]
NO_TIE = "_none"
TIE_KEY = {"conf_w": scorers.TIE_KEY["conf_w"], "subject_mean": NO_TIE, "within_subject": "conf_w",
           "disagree": scorers.TIE_KEY["disagree"], "length": scorers.TIE_KEY["length"]}
LABEL = "Exploratory, post-unblinding. No decision rule."


def subject_scores(sc: pd.DataFrame) -> pd.DataFrame:
    """Add subject_mean and within_subject from conf_w. Reads only item_id, subject and conf_w."""
    out = sc.copy()
    out["subject_mean"] = out.groupby("subject")["conf_w"].transform("mean")
    out["within_subject"] = out["conf_w"] - out["subject_mean"]
    out[NO_TIE] = 0.0
    return out


def per_subject(t: pd.DataFrame, y: np.ndarray, budget: float) -> list[dict]:
    """Items, positives, expected flagged items and recall at the budget, per subject, for each scorer."""
    k = rank.budget_k(len(t), budget)
    incl = {s: rank.inclusion(t[s].to_numpy(float), t[TIE_KEY[s]].to_numpy(float), k) for s in NAMES}
    rows = []
    for subj in sorted(t["subject"].unique()):
        m = (t["subject"] == subj).to_numpy()
        pos = int(y[m].sum())
        r = {"subject": subj, "items": int(m.sum()), "positives": pos, "flagged": {}, "recall": {}}
        for s in NAMES:
            r["flagged"][s] = float(incl[s][m].sum())
            found = float((incl[s][m] * y[m]).sum())
            w = stats.wilson(found, pos)
            if w is not None:
                w["k"] = found
            r["recall"][s] = w
        rows.append(r)
    return sorted(rows, key=lambda r: (-r["positives"], r["subject"]))


def _block(t: pd.DataFrame) -> dict:
    y = t["flaw_primary"].to_numpy(float)
    S = {s: (t[s].to_numpy(float), t[TIE_KEY[s]].to_numpy(float)) for s in NAMES}
    return evaluate.evaluate_scorers(y, S, evaluate.boot_indices(len(t)), PAIRS)


def run() -> dict:
    f = subject_scores(evaluate.frame(caller="itemtriage.explore"))
    test = f[(f["fold"] == "test") & f["flaw_primary"].notna()]
    nv = test[test["subject"] != "virology"]
    out = {"label": LABEL, "class": "primary", "budget": evaluate.PRIMARY_BUDGET, "scorers": list(NAMES),
           "tie_keys": {k: (None if v == NO_TIE else v) for k, v in TIE_KEY.items()},
           "all": _block(test), "virology_excluded": _block(nv),
           "per_subject": per_subject(test, test["flaw_primary"].to_numpy(float), evaluate.PRIMARY_BUDGET),
           "memorization_test": evaluate.memorization(f, folds=("test",))}
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out
