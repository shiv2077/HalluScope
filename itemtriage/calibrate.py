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
LTT_START_COVERAGE = 0.10  # deviation, see ltt_from_coverage
N_GRID = 200
DELTA = 0.10
COVERAGES = (0.5, 0.7, 0.9)
CAL_OUT = paths.RESULTS / "calibration.json"


def _hash(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def ltt_from_coverage(scores: np.ndarray, flawed: np.ndarray, alpha: float, delta: float = DELTA,
                      start: float = LTT_START_COVERAGE, n_grid: int = N_GRID) -> dict:
    """Fixed-sequence Learn-Then-Test with a Hoeffding bound, over thresholds at coverage start..1.

    Declared post-unblinding deviation. routing.calibrate.calibrate_risk starts its sequence at the single
    lowest score, where one accepted item gives a Hoeffding bound above 1, so its first test always fails and
    it can never return a threshold. Starting at 10% coverage keeps the sequence fixed in advance (coverage
    targets do not depend on labels) while letting the bound be informative. FWER control is unchanged.
    """
    ok = np.isfinite(scores)
    s, y = scores[ok], flawed[ok].astype(float)
    best = None
    for q in np.linspace(start, 1.0, n_grid):
        t = float(np.quantile(s, q))
        acc = s <= t
        n = int(acc.sum())
        risk = float(y[acc].mean())
        ucb = stats.hoeffding_ucb(risk, n, delta)
        if ucb <= alpha:
            best = {"threshold": t, "coverage_cal": n / len(s), "risk_cal": risk, "ucb": ucb, "n_accepted": n}
        else:
            break
    return best or {"threshold": None, "reason": f"no threshold from {start:.0%} coverage has Hoeffding UCB <= {alpha}"}


def run() -> dict:
    evaluate.prereg_commit()
    f = evaluate.frame()
    irt_k = evaluate.select_irt(f)["chosen"]
    names = ["conf_w", "conf", "disagree", irt_k, "ptrue_key", "length", "dup_options"]
    names = evaluate.available(f, names)
    cal = f[f["fold"].isin(["calibration_fit", "validation"]) & f["flaw_primary"].notna()]
    test_all = f[f["fold"] == "test"]
    test = test_all[test_all["flaw_primary"].notna()]
    out = {"scores_hash": _hash(scorers.SCORES), "labels_hash": _hash(paths.LABELS), "n_calibration": int(len(cal)),
           "calibration_positives": int(cal["flaw_primary"].sum()), "n_test": int(len(test_all)), "fixed_cutoffs": {},
           "risk": {}, "risk_deviation": {}, "required_n": {},
           "risk_note": "risk is the preregistered routing.calibrate.calibrate_risk, which cannot qualify by construction; "
                        "risk_deviation is the declared post-unblinding fix that starts the fixed sequence at 10% coverage"}
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
        dev = {}
        for a in ALPHAS:
            r = ltt_from_coverage(s_cal, y_cal, a)
            if r.get("threshold") is not None:
                acc = test[k].to_numpy(float) <= r["threshold"]
                yt = test["flaw_primary"].to_numpy(float)
                r = {**r, "test_coverage": stats.wilson(int(acc.sum()), len(acc)),
                     "test_risk": stats.wilson(int(yt[acc].sum()), int(acc.sum()))}
            dev[str(a)] = r
        out["risk_deviation"][k] = dev
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
