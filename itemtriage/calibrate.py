"""H3: fixed cutoffs from calibration against the batch-quantile budget policy, and the Learn-Then-Test risk lane.

Calibration is calibration fit plus validation, known items only. Fixed cutoffs are the 5%, 10% and 20% score
quantiles there (routing.calibrate.calibrate_budget). On test, the realized flag rate of each cutoff gets a
Wilson interval; the budget policy flags exactly ceil(budget x n) by construction. Learn-Then-Test
(routing.calibrate.calibrate_risk, Hoeffding, delta 0.10) accepts items with score at or below the threshold,
with risk the primary flaw rate among accepted items. Writes results/itemtriage/calibration.json.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from routing import calibrate as rcal
from routing import stats

from . import evaluate, paths, rank, scorers

ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
COVERAGES = (0.5, 0.7, 0.9)
CAL_OUT = paths.RESULTS / "calibration.json"


def _hash(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run() -> dict:
    evaluate.prereg_commit()
    f = evaluate.frame()
    irt_k = evaluate.select_irt(f)["chosen"]
    names = ["conf_w", "conf", "disagree", irt_k, "ptrue_key", "length", "dup_options"]
    names = [n for n in names if f[n].notna().all()]
    cal = f[f["fold"].isin(["calibration_fit", "validation"]) & f["flaw_primary"].notna()]
    test_all = f[f["fold"] == "test"]
    test = test_all[test_all["flaw_primary"].notna()]
    out = {"scores_hash": _hash(scorers.SCORES), "labels_hash": _hash(paths.LABELS), "n_calibration": int(len(cal)),
           "calibration_positives": int(cal["flaw_primary"].sum()), "n_test": int(len(test_all)), "fixed_cutoffs": {},
           "risk": {}, "required_n": {}}
    for k in names:
        s_cal, s_test = cal[k].to_numpy(float), test_all[k].to_numpy(float)
        rows = {}
        for b in evaluate.BUDGETS:
            cut = rcal.calibrate_budget(s_cal, b)
            flagged = int((s_test >= cut).sum())
            rows[str(b)] = {"cutoff": float(cut), "test_flag_rate": stats.wilson(flagged, len(s_test)),
                            "budget_policy_flag_rate": rank.budget_k(len(s_test), b) / len(s_test)}
        out["fixed_cutoffs"][k] = rows
        y_cal = cal["flaw_primary"].to_numpy(float)
        risk = {}
        for a in ALPHAS:
            r = rcal.calibrate_risk(s_cal, y_cal, a, DELTA)
            if r.get("threshold") is not None:
                acc = test[k].to_numpy(float) <= r["threshold"]
                yt = test["flaw_primary"].to_numpy(float)
                r = {**r, "test_coverage": stats.wilson(int(acc.sum()), len(acc)),
                     "test_risk": stats.wilson(int(yt[acc].sum()), int(acc.sum()))}
            risk[str(a)] = r
        out["risk"][k] = risk
        order = np.argsort(s_cal, kind="stable")
        ys = y_cal[order]
        req = {}
        for cov in COVERAGES:
            n_acc = int(round(cov * len(ys)))
            r_hat = float(ys[:n_acc].mean())
            req[str(cov)] = {"risk_hat_at_coverage": r_hat,
                             "required_n": {str(a): rcal.required_n(a, DELTA, cov, r_hat) for a in ALPHAS}}
        out["required_n"][k] = req
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    CAL_OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out
