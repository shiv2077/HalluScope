"""Evaluation on the test split only. Never computes a calibrated cutoff.

Two policy families (routing/policy.py):
  budget policy: flag the top ceil(budget * n) rows of the batch by batch quantile. Every review-lift
                 number uses this, and the bootstrap re-applies the quantile inside each resample.
  risk policy:   apply a cutoff from results/calibration.json; the flag rate floats. Used for the
                 cascade's risk-policy points and for the cutoff drift subsection.

Outputs: figures/review_lift.png, figures/cascade_frontier.png, figures/risk_coverage.png,
results/evaluation.json, reports/routing_report.md. Every number carries an interval.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import (  # noqa: E402
    calibrate,
    cost,
    data,
    gates,
    prevalence,  # noqa: E402
    stats,
)
from .policy import batch_cutoff, flag_by_budget, flag_by_risk  # noqa: E402
from .scorers import EXTRA_CALLS  # noqa: E402

BUDGETS = calibrate.BUDGETS
ALPHAS = calibrate.ALPHAS
COLORS = {"mean_logprob": "#1f77b4", "min_k_logprob": "#ff7f0e", "length": "#7f7f7f",
          "length_residual": "#2ca02c", "consistency_k3": "#d62728", "p_true": "#9467bd"}


# ------------------------------------------------------------------ data assembly
def assemble() -> dict:
    d = data.load_split("test")
    cfg = data.models_config()
    s = data.read_scores("test")
    c = d["consensus"]
    cheap = c[c["model"] == cfg["cheap"]][["qid", "label"]].rename(columns={"label": "label_cheap"})
    strong = c[c["model"] == cfg["strong"]][["qid", "label"]].rename(columns={"label": "label_strong"})
    m = s.merge(cheap, on="qid", how="left").merge(strong, on="qid", how="left")
    g = d["generations"]
    g["cost"] = [cost.cost_usd(r.model, r.input_tokens, r.output_tokens) for r in g.itertuples()]
    per_role = g.pivot_table(index="qid", columns="role", values="cost", aggfunc="sum").reset_index()
    m = m.merge(per_role[["qid", "primary", "strong"]].rename(columns={"primary": "cost_cheap", "strong": "cost_strong"}),
                on="qid", how="left")
    m["extra_cost"] = m["extra_cost_usd"].map(json.loads)
    m = m.sort_values("qid").reset_index(drop=True)
    return {"df": m, "n_total": len(m), "cfg": cfg}


def _headline_rows(m: pd.DataFrame) -> pd.DataFrame:
    return m[m["label_cheap"].isin(["CORRECT", "INCORRECT"])].copy()


def _flags(s: np.ndarray, budget: float) -> np.ndarray:
    """Budget-policy flags on a numpy score vector. Positional index is the tie-breaker."""
    return flag_by_budget(pd.Series(s), budget).to_numpy()


def _catch(y, s, b):
    fl = _flags(s, b)
    return None if y.sum() == 0 or fl.sum() == 0 else float(y[fl].sum() / y.sum())


def _lift(y, s, b):
    fl = _flags(s, b)
    return None if y.sum() == 0 or fl.sum() == 0 else float((y[fl].sum() / y.sum()) / (fl.sum() / len(fl)))


def _residual(y, s, b):
    fl = _flags(s, b)
    return None if (~fl).sum() == 0 else float(y[~fl].mean())


# ------------------------------------------------------------------ 1. review lift (budget policy)
def review_lift(m: pd.DataFrame) -> dict:
    out = {}
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    out["_n"] = int(len(h))
    out["_error_rate"] = stats.wilson(int(y_all.sum()), len(h))
    for sc in data.SCORERS:
        s = h[sc].astype(float).to_numpy()
        ok = np.isfinite(s)
        y, s = y_all[ok], s[ok]
        out[sc] = {}
        for b in BUDGETS:
            fl = _flags(s, b)
            out[sc][str(b)] = {"policy": "budget", "cutoff_batch": float(s[fl].min()) if fl.any() else None,
                               "flagged_fraction": stats.wilson(int(fl.sum()), len(fl)),
                               "precision": stats.wilson(int(y[fl].sum()), int(fl.sum())),
                               "catch_rate": stats.boot(lambda yy, ss: _catch(yy, ss, b), y, s),
                               "random_catch_rate": stats.wilson(int(fl.sum()), len(fl)),
                               "lift": stats.boot(lambda yy, ss: _lift(yy, ss, b), y, s),
                               "residual_error_rate": stats.boot(lambda yy, ss: _residual(yy, ss, b), y, s)}
    return out


def fig_review_lift(lift: dict, fl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sc in data.SCORERS:
        xs = [lift[sc][str(b)]["flagged_fraction"]["est"] for b in BUDGETS]
        ys = [lift[sc][str(b)]["catch_rate"] for b in BUDGETS]
        if any(v is None for v in ys):
            continue
        style = "--" if sc == "length" else "-"
        label = "length (control)" if sc == "length" else sc + (" (not distinguishable from length)" if fl.get(sc) else "")
        ax.plot(xs, [v["est"] for v in ys], style, marker="x" if fl.get(sc) else "o", color=COLORS[sc], label=label)
        ax.fill_between(xs, [v["lo"] for v in ys], [v["hi"] for v in ys], color=COLORS[sc], alpha=0.12)
    ax.plot([0, 0.35], [0, 0.35], ":", color="black", label="random")
    ax.set_xlabel("review budget (fraction of the batch flagged, budget policy)")
    ax.set_ylabel("catch rate (fraction of errors inside flagged slice)")
    ax.set_title("Review lift on the test split, budget policy (bands: 95% bootstrap)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(data.FIGURES / "review_lift.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------ 1a. residual error against budget
CURVE_BUDGETS = tuple(round(0.05 * i, 2) for i in range(1, 11))


def residual_curve(m: pd.DataFrame) -> dict:
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    out = {"budgets": list(CURVE_BUDGETS), "base_error": stats.wilson(int(y_all.sum()), len(y_all))}
    for sc in data.SCORERS:
        s = h[sc].astype(float).to_numpy()
        ok = np.isfinite(s)
        y, s = y_all[ok], s[ok]
        out[sc] = [stats.boot(lambda yy, ss, b=b: _residual(yy, ss, b), y, s) for b in CURVE_BUDGETS]
    return out


def fig_residual(rc: dict, fl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = rc["budgets"]
    for sc in data.SCORERS:
        pts = rc[sc]
        if any(p is None for p in pts):
            continue
        label = "length (control)" if sc == "length" else sc + (" (not distinguishable from length)" if fl.get(sc) else "")
        ax.plot(xs, [p["est"] for p in pts], "--" if sc == "length" else "-", marker="x" if fl.get(sc) else "o", ms=3, color=COLORS[sc], label=label)
        ax.fill_between(xs, [p["lo"] for p in pts], [p["hi"] for p in pts], color=COLORS[sc], alpha=0.10)
    ax.axhline(rc["base_error"]["est"], color="black", ls=":", label="random (base error rate)")
    ax.axhline(rc["base_error"]["est"] / 2, color="black", lw=0.6, ls="-.", label="half the base error rate")
    ax.set_xlabel("review budget (budget policy)")
    ax.set_ylabel("residual error among unreviewed rows")
    ax.set_title("Residual error against budget on the test split (bands: 95% bootstrap)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(data.FIGURES / "residual_error.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


def residual_render(rc: dict, lift: dict, best: str) -> list[str]:
    base = rc["base_error"]
    half = base["est"] / 2
    reached = next((b for b, p in zip(rc["budgets"], rc[best]) if p is not None and p["est"] <= half), None)
    L = ["", "### Residual error against budget", "", "![residual error](../figures/residual_error.png)", "",
         f"Base error rate with no review: {_pct(base)}. Budget policy from 0.05 to 0.50.", "",
         "| scorer | " + " | ".join(f"{b:.2f}" for b in rc["budgets"]) + " |", "|---|" + "---|" * len(rc["budgets"])]
    for sc in data.SCORERS:
        L.append(f"| {sc} | " + " | ".join(_pct(p) for p in rc[sc]) + " |")
    if reached is None:
        L += ["", f"`{best}` does not bring residual error to half the base rate ({100 * half:.1f}%) by a 0.50 budget; "
              f"at 0.50 it reaches {_pct(rc[best][-1])}."]
    else:
        i = rc["budgets"].index(reached)
        L += ["", f"`{best}` reaches half the base error rate ({100 * half:.1f}%) at a budget of {reached:.2f}, where residual error is {_pct(rc[best][i])}."]
    return L


# ------------------------------------------------------------------ 1b. cutoff drift
def cutoff_drift(m: pd.DataFrame, cal: dict) -> dict:
    h = _headline_rows(m)
    out = {}
    for sc in data.SCORERS:
        s = h[sc].astype(float)
        s.index = h["qid"]
        out[sc] = {}
        for b in BUDGETS:
            cut = cal["budgets"][sc][str(b)]
            fl = flag_by_risk(s, cut).to_numpy()
            out[sc][str(b)] = {"cutoff_calibration": cut, "flag_rate_test": stats.wilson(int(fl.sum()), len(fl)),
                               "cutoff_batch": batch_cutoff(s, b)}
    return out


# ------------------------------------------------------------------ 2. cascade frontier
def _cascade_point(esc, yc, ys, cc, cs, extra, cutoff, policy):
    y = np.where(esc, ys, yc)
    c = cc + extra + np.where(esc, cs, 0.0)
    fix_n = int(((yc == 1) & esc).sum())
    fix_k = int(((yc == 1) & esc & (ys == 0)).sum())
    brk_n = int(((yc == 0) & esc).sum())
    brk_k = int(((yc == 0) & esc & (ys == 1)).sum())
    def gap(yc_, ys_, y_):
        d = yc_.mean() - ys_.mean()
        return None if d <= 0 else float((yc_.mean() - y_.mean()) / d)

    def cfrac(cc_, cs_, c_):
        d = cs_.mean() - cc_.mean()
        return None if d <= 0 else float((c_.mean() - cc_.mean()) / d)

    return {"policy": policy, "cutoff": cutoff, "escalated_fraction": stats.wilson(int(esc.sum()), len(esc)),
            "error": stats.boot(lambda yy: float(yy.mean()), y),
            "cost_per_1000": stats.boot(lambda x: float(x.mean()) * 1000, c),
            "gap_closed": stats.boot(gap, yc, ys, y), "cost_fraction": stats.boot(cfrac, cc, cs, c),
            "fix_rate": stats.wilson(fix_k, fix_n), "break_rate": stats.wilson(brk_k, brk_n)}


def cascade(m: pd.DataFrame, cal: dict) -> dict:
    both = m[m["label_cheap"].isin(["CORRECT", "INCORRECT"]) & m["label_strong"].isin(["CORRECT", "INCORRECT"])].copy()
    yc = (both["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    ys = (both["label_strong"] == "INCORRECT").astype(int).to_numpy()
    cc = both["cost_cheap"].to_numpy(float)
    cs = both["cost_strong"].to_numpy(float)
    out = {"_n": int(len(both)), "_n_excluded": int(len(m) - len(both)),
           "all_cheap": {"error": stats.wilson(int(yc.sum()), len(yc)), "cost_per_1000": stats.boot(lambda c: float(c.mean()) * 1000, cc)},
           "all_strong": {"error": stats.wilson(int(ys.sum()), len(ys)), "cost_per_1000": stats.boot(lambda c: float(c.mean()) * 1000, cs)},
           "scorers": {}}
    for sc in data.SCORERS:
        s = both[sc].astype(float).to_numpy()
        extra = np.array([e[sc] for e in both["extra_cost"]], float)
        points = {}
        for b in BUDGETS:
            esc = _flags(s, b)
            points[f"budget_{b}"] = _cascade_point(esc, yc, ys, cc, cs, extra, float(s[esc].min()) if esc.any() else None, "budget")
        for b in BUDGETS:
            cut = cal["budgets"][sc][str(b)]
            esc = np.isfinite(s) & (s >= cut)
            points[f"risk_calcut_{b}"] = _cascade_point(esc, yc, ys, cc, cs, extra, cut, "risk")
        for a in ALPHAS:
            r = cal["risk"][sc][str(a)]
            if r.get("threshold") is not None:
                esc = np.isfinite(s) & (s > r["threshold"])
                points[f"risk_alpha_{a}"] = _cascade_point(esc, yc, ys, cc, cs, extra, r["threshold"], "risk")
        out["scorers"][sc] = points
    return out


def fig_cascade(cas: dict, fl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for sc, pts in cas["scorers"].items():
        for fam, marker, ls in (("budget", "o", "-"), ("risk", "^", ":")):
            sel = [p for k, p in pts.items() if p["policy"] == fam]
            if not sel:
                continue
            xs = [p["cost_per_1000"]["est"] for p in sel]
            ys = [p["error"]["est"] for p in sel]
            ye = [[p["error"]["est"] - p["error"]["lo"] for p in sel], [p["error"]["hi"] - p["error"]["est"] for p in sel]]
            base = "length (control)" if sc == "length" else sc + (" (not distinguishable from length)" if fl.get(sc) else "")
            label = f"{base}, {fam} policy"
            ax.errorbar(xs, ys, yerr=ye, fmt=marker + ls, color=COLORS[sc], label=label, capsize=2, alpha=0.9, ms=6)
    for name, mk in (("all_cheap", "s"), ("all_strong", "D")):
        p = cas[name]
        ax.plot(p["cost_per_1000"]["est"], p["error"]["est"], mk, ms=11, color="black", label=name.replace("_", " "))
    ax.set_xlabel("cost per 1000 queries (USD, config/pricing.json)")
    ax.set_ylabel("error rate after routing")
    ax.set_title("Cascade frontier on test: circles budget policy, triangles risk policy")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(data.FIGURES / "cascade_frontier.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------ 2b. cost tiers
def cost_tiers(m: pd.DataFrame, lift: dict) -> list:
    h = _headline_rows(m)
    rows = []
    for sc in data.SCORERS:
        extra = np.array([e[sc] for e in h["extra_cost"]], float)
        rows.append({"scorer": sc, "extra_calls": EXTRA_CALLS[sc],
                     "extra_cost_per_1000": stats.boot(lambda x: float(x.mean()) * 1000, extra),
                     "lift_0.2": lift[sc]["0.2"]["lift"]})
    return sorted(rows, key=lambda r: (r["extra_cost_per_1000"]["est"], r["scorer"]))


# ------------------------------------------------------------------ 3. risk-coverage
def _aurc(y, s):
    order = np.argsort(s, kind="stable")
    ys = y[order]
    risks = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(risks.mean())


def risk_coverage(m: pd.DataFrame) -> dict:
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    out = {"_n": int(len(h))}
    for sc in data.SCORERS:
        s = h[sc].astype(float).to_numpy()
        ok = np.isfinite(s)
        y, s = y_all[ok], s[ok]
        oracle = _aurc(y, y.astype(float))
        order = np.argsort(s, kind="stable")
        ys = y[order]
        curve_risk = (np.cumsum(ys) / np.arange(1, len(ys) + 1)).tolist()
        curve_cov = (np.arange(1, len(ys) + 1) / len(ys)).tolist()
        step = max(1, len(ys) // 60)
        out[sc] = {"aurc": stats.boot(_aurc, y, s), "oracle_aurc": oracle,
                   "curve": {"coverage": curve_cov[::step], "risk": curve_risk[::step]}}
    return out


def fig_risk_coverage(rc: dict, fl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sc in data.SCORERS:
        c = rc[sc]["curve"]
        label = "length (control)" if sc == "length" else sc + (" (not distinguishable from length)" if fl.get(sc) else "")
        ax.plot(c["coverage"], c["risk"], "--" if sc == "length" else "-", color=COLORS[sc], label=label,
                marker="x" if fl.get(sc) else None, ms=3)
    ax.set_xlabel("coverage (fraction of rows returned without escalation)")
    ax.set_ylabel("selective risk (error rate among returned rows)")
    ax.set_title("Risk-coverage on the test split")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(data.FIGURES / "risk_coverage.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------ 4. length stratification (budget policy per tertile)
def length_strata(m: pd.DataFrame) -> dict:
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    L = h["length"].to_numpy(float)
    edges = np.quantile(L, [1 / 3, 2 / 3])
    tert = np.digitize(L, edges)
    out = {"edges": edges.tolist(), "n_per_tertile": [int((tert == t).sum()) for t in range(3)]}
    for sc in data.SCORERS:
        s_all = h[sc].astype(float).to_numpy()
        out[sc] = []
        for t in range(3):
            sel = (tert == t) & np.isfinite(s_all)
            y, s = y_all[sel], s_all[sel]
            fl = _flags(s, 0.2)
            out[sc].append({"tertile": t, "n": int(sel.sum()), "errors": int(y.sum()), "flagged": int(fl.sum()),
                            "lift": stats.boot(lambda yy, ss: _lift(yy, ss, 0.2), y, s)})
    return out


# ------------------------------------------------------------------ 5. conformal
def conformal(m: pd.DataFrame, cal: dict) -> dict:
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    out = {}
    for sc in data.SCORERS:
        s = h[sc].astype(float).to_numpy()
        out[sc] = {}
        for a in ALPHAS:
            r = cal["risk"][sc][str(a)]
            if r.get("threshold") is None:
                out[sc][str(a)] = {"threshold": None, "reason": r["reason"]}
                continue
            acc = np.isfinite(s) & (s <= r["threshold"])
            out[sc][str(a)] = {"threshold": r["threshold"], "coverage_cal": r["coverage_cal"],
                               "coverage_test": stats.wilson(int(acc.sum()), len(acc)),
                               "risk_test": stats.wilson(int(y_all[acc].sum()), int(acc.sum()))}
    return out


# ------------------------------------------------------------------ report
def _pct(d, k=1):
    if d is None:
        return "n/a"
    return f"{100 * d['est']:.{k}f}% [{100 * d['lo']:.{k}f}, {100 * d['hi']:.{k}f}]"


def _usd(d):
    return "n/a" if d is None else f"${d['est']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def best_scorer(lift: dict, fl: dict) -> str:
    cands = [(lift[sc]["0.2"]["lift"]["est"], sc) for sc in data.SCORERS
             if sc != "length" and lift[sc]["0.2"]["lift"] is not None]
    unflagged = [c for c in cands if not fl.get(c[1])]
    return max(unflagged or cands)[1]


def best_free_scorer(lift: dict, fl: dict) -> str:
    cands = [(lift[sc]["0.2"]["lift"]["est"], sc) for sc in data.SCORERS
             if sc != "length" and EXTRA_CALLS[sc] == 0 and lift[sc]["0.2"]["lift"] is not None]
    unflagged = [c for c in cands if not fl.get(c[1])]
    return max(unflagged or cands)[1]


def _star(sc, fl):
    return " *" if fl.get(sc) and sc != "length" else ""


def render_report(ev: dict, g: dict, cal: dict, fl: dict, calls: dict, cfg: dict) -> str:
    lift, cas, rc, ls, cf = ev["review_lift"], ev["cascade"], ev["risk_coverage"], ev["length_strata"], ev["conformal"]
    drift, tiers = ev["cutoff_drift"], ev["cost_tiers"]
    best, free = ev["best_scorer"], ev["best_free_scorer"]
    b20 = lift[best]["0.2"]
    pts = cas["scorers"][best]
    bp_name = min((k for k in pts if pts[k]["policy"] == "budget"), key=lambda k: pts[k]["error"]["est"])
    bp = pts[bp_name]
    tag = " (not distinguishable from length)" if fl.get(best) else ""
    L = ["# Routing report", "",
         f"Prices from config/pricing.json (priced_on {calls['priced_on']}). Brackets are 95 percent bootstrap "
         "intervals (2000 resamples, seed 42) or Wilson intervals for proportions. Raw counts carry no interval. "
         "Review lift, length strata and the cascade's budget points use the budget policy: the top ceil(budget * n) rows "
         "of the test batch by batch quantile, recomputed inside every bootstrap resample. Cascade risk points and the "
         "cutoff drift subsection use fixed cutoffs from results/calibration.json.", "",
         "## Summary", "",
         f"On {lift['_n']} test rows with a consensus label, the cheap model ({cfg['cheap']}) has an error rate of {_pct(lift['_error_rate'])}. "
         f"Under the budget policy at 20 percent, the best scorer is `{best}`{tag}, catching {_pct(b20['catch_rate'])} of errors "
         f"against {_pct(b20['random_catch_rate'])} for random, a lift of {stats.f(b20['lift'], 2)}; the best free scorer is `{free}` "
         f"at lift {stats.f(lift[free]['0.2']['lift'], 2)}. "
         f"Residual error among unreviewed rows at that budget is {_pct(b20['residual_error_rate'])} for `{best}`. "
         f"The cascade at the `{best}` {bp_name.replace('_', ' ')} point has error {_pct(bp['error'])} at {_usd(bp['cost_per_1000'])} per thousand queries, "
         f"against all-cheap {_pct(cas['all_cheap']['error'])} at {_usd(cas['all_cheap']['cost_per_1000'])} "
         f"and all-strong {_pct(cas['all_strong']['error'])} at {_usd(cas['all_strong']['cost_per_1000'])}. "
         f"Fixed cutoffs from the {cal['n']}-row calibration split are reported separately as risk-policy points and in the cutoff drift table.", "",
         "## Data and labels", "",
         f"- Calibration: {cal['n']} rows with a consensus label ({cal['positives']} INCORRECT), {cal['n_ambiguous_excluded']} AMBIGUOUS excluded. "
         f"Scores file hash {cal['calibration_hash']}, labels hash {cal['labels_hash']}.",
         f"- Test: {ev['n_total']} questions, {lift['_n']} with a consensus cheap-model label, {ev['n_total'] - lift['_n']} AMBIGUOUS excluded. "
         f"Cascade uses the {cas['_n']} rows where both models have a consensus label ({cas['_n_excluded']} excluded).",
         f"- Judges: `{cfg['judge_a']}` and `{cfg['judge_b']}`, blind to every score, consensus rule in routing/data.py. "
         "Agreement and kappa in results/phase1_notes.md.", "",
         "## Validity gates", "", f"See results/gates_test.md. Template rows: {g['template_rows']}. Split disjoint: {g['disjoint']}. "
         f"Power at actual n: {g['power_at_actual_n']:.2f}." if g["power_at_actual_n"] is not None else "See results/gates_test.md.", "",
         "| scorer | AUROC | vs length p | flag |", "|---|---|---|---|"]
    for sc, r in g["scorers"].items():
        vl = r.get("vs_length")
        flag = "control" if sc == "length" else ("" if r["beats_length"] else "not distinguishable from length")
        L.append(f"| {sc} | {stats.f(r['auroc'])} | {vl['p_delong']:.3f} | {flag} |" if vl else f"| {sc} | {stats.f(r['auroc'])} | n/a | control |")
    L += ["", "## Review lift", "",
          "Budget policy: the top ceil(budget * n) rows of the test batch. Random-baseline catch rate equals the exact flagged fraction.", "",
          "![review lift](../figures/review_lift.png)", "",
          "| scorer | budget | flagged | catch rate | random | lift | precision | residual error |", "|---|---:|---|---|---|---|---|---|"]
    for sc in data.SCORERS:
        for b in BUDGETS:
            r = lift[sc][str(b)]
            L.append(f"| {sc}{_star(sc, fl)} | {b:.2f} | {_pct(r['flagged_fraction'])} | {_pct(r['catch_rate'])} | "
                     f"{_pct(r['random_catch_rate'])} | {stats.f(r['lift'], 2)} | {_pct(r['precision'])} | {_pct(r['residual_error_rate'])} |")
    L += ["", "`*` marks a scorer not distinguishable from length. `length` is the control. Precision is the fraction of flagged rows "
          "that are INCORRECT, which is what a reviewer sees. Residual error is the error rate among unflagged rows, assuming review fixes what it finds.", ""]
    L += residual_render(ev["residual_curve"], lift, best)
    L += ["",
          "### Cutoff drift", "",
          "For each scorer and budget: the cutoff a calibration-derived quantile would give, the flag rate that fixed cutoff "
          "actually produces on test, and the batch-quantile cutoff the budget policy used on test instead. "
          "A cutoff fixed on one draw does not deliver a fixed flag rate on another, which is why budget policies use batch quantiles.", "",
          "| scorer | budget | calibration cutoff | flag rate on test at that cutoff | batch cutoff on test |", "|---|---:|---:|---|---:|"]
    for sc in data.SCORERS:
        for b in BUDGETS:
            r = drift[sc][str(b)]
            L.append(f"| {sc} | {b:.2f} | {r['cutoff_calibration']:.4f} | {_pct(r['flag_rate_test'])} | {r['cutoff_batch']:.4f} |")
    L += ["", "### Cost tiers", "",
          "What each scorer costs per score beyond the primary answer, from logged tokens and config/pricing.json, and what it buys "
          "at the 0.20 budget. Sorted by extra cost.", "",
          "| scorer | extra calls | extra cost per 1000 | lift at 0.20 |", "|---|---:|---|---|"]
    for r in tiers:
        L.append(f"| {r['scorer']}{_star(r['scorer'], fl)} | {r['extra_calls']} | {_usd(r['extra_cost_per_1000'])} | {stats.f(r['lift_0.2'], 2)} |")
    L += prevalence.render(ev["prevalence"], best, fl)
    L += ["", "## Cascade frontier", "", "![cascade frontier](../figures/cascade_frontier.png)", "",
          f"Endpoints: all cheap error {_pct(cas['all_cheap']['error'])} at {_usd(cas['all_cheap']['cost_per_1000'])} per 1000; "
          f"all strong error {_pct(cas['all_strong']['error'])} at {_usd(cas['all_strong']['cost_per_1000'])} per 1000. "
          "The strong model's error rate is the ceiling of any cascade. Budget points (circles) escalate the top ceil(budget * n) rows of the batch; "
          "risk points (triangles) escalate rows at or above a fixed cutoff from results/calibration.json, so their escalated fraction floats.", "",
          "| scorer | policy | point | cutoff | escalated | error | cost per 1000 | gap closed | cost fraction | fix rate | break rate |",
          "|---|---|---|---:|---|---|---|---|---|---|---|"]
    for sc, pts in cas["scorers"].items():
        for name, p in pts.items():
            cut = "n/a" if p["cutoff"] is None else f"{p['cutoff']:.4f}"
            L.append(f"| {sc}{_star(sc, fl)} | {p['policy']} | {name} | {cut} | {_pct(p['escalated_fraction'])} | {_pct(p['error'])} | "
                     f"{_usd(p['cost_per_1000'])} | {_pct(p['gap_closed'])} | {_pct(p['cost_fraction'])} | {_pct(p['fix_rate'])} | {_pct(p['break_rate'])} |")
    L += ["", "Fix rate: among escalated rows the cheap model got wrong, the fraction the strong model got right. "
          "Break rate: among escalated rows the cheap model got right, the fraction the strong model got wrong. "
          "Gap closed: (error all cheap minus error at point) over (error all cheap minus error all strong). "
          "Cost fraction: (cost at point minus cost all cheap) over (cost all strong minus cost all cheap). Both bootstrapped jointly over rows. "
          "`risk_calcut_b` is the fixed cutoff that flagged the top b fraction on calibration; "
          "`risk_alpha_a` points appear only when calibrate_risk finds a threshold.", "",
          "## Risk-coverage", "", "![risk coverage](../figures/risk_coverage.png)", "",
          "| scorer | AURC | oracle AURC |", "|---|---|---:|"]
    for sc in data.SCORERS:
        L.append(f"| {sc} | {stats.f(rc[sc]['aurc'])} | {rc[sc]['oracle_aurc']:.3f} |")
    L += ["", "## Length", "", f"Length tertile edges (output tokens): {ls['edges'][0]:.0f}, {ls['edges'][1]:.0f}. "
          f"Rows per tertile: {ls['n_per_tertile']}. Budget-policy lift at 0.20 applied within each tertile.", "",
          "| scorer | short | medium | long |", "|---|---|---|---|"]
    for sc in data.SCORERS:
        L.append(f"| {sc} | " + " | ".join(f"{stats.f(t['lift'], 2)} (n={t['n']}, err={t['errors']})" for t in ls[sc]) + " |")
    L += ["", "## Conformal and required n", "",
          f"Learn-Then-Test with a Hoeffding bound at delta {calibrate.DELTA}, calibrated on n={cal['n']}.", "",
          "| scorer | alpha | threshold | coverage (cal) | coverage (test) | selective risk (test) |", "|---|---:|---|---|---|---|"]
    for sc in data.SCORERS:
        for a in ALPHAS:
            r = cf[sc][str(a)]
            if r["threshold"] is None:
                L.append(f"| {sc} | {a} | none qualifies | | | |")
            else:
                L.append(f"| {sc} | {a} | {r['threshold']:.4f} | {100 * r['coverage_cal']:.1f}% | {_pct(r['coverage_test'])} | {_pct(r['risk_test'])} |")
    L += ["", "Required calibration size for a threshold to pass the test (n_acc = ln(1/delta) / (2 (alpha - risk_hat)^2), divided by target coverage):", "",
          "| alpha | target coverage | n if risk_hat = 0 | n if risk_hat = alpha/2 |", "|---:|---:|---:|---:|"]
    for r in cal["required_n"]:
        L.append(f"| {r['alpha']} | {r['target_coverage']} | {r['n_if_risk_hat_0']} | {r['n_if_risk_hat_half_alpha']} |")
    L += ["", "## Cost table", "", f"From results/calls.jsonl. Total calls {calls['n_calls']}, total cost ${calls['cost_usd']:.4f}.", "",
          "| purpose | calls | input tokens | output tokens | USD |", "|---|---:|---:|---:|---:|"]
    for k in sorted(calls["by_purpose"]):
        b = calls["by_purpose"][k]
        L.append(f"| {k} | {b['n_calls']} | {b['input_tokens']} | {b['output_tokens']} | {b['cost_usd']:.4f} |")
    hp = data.RESULTS / "human_agreement.md"
    if hp.exists():
        demoted = "\n".join(("##" + ln) if ln.startswith("#") else ln for ln in hp.read_text().splitlines())
        L += ["", "## Human labels", "", "Included verbatim from results/human_agreement.md (headings demoted two levels).", "", demoted]
    L += ["", "## Limitations", "",
          "- Short-answer QA (HaluEval) only. Long multi-claim outputs need claim-level scoring (docs/ROADMAP.md).",
          "- OpenAI models only for generation. Logprob scorers need a provider that returns logprobs; consistency and P(True) do not.",
          "- Labels are from two LLM judges from one provider, not humans. AMBIGUOUS rows are excluded and counted above.",
          f"- Conformal thresholds are calibrated on {cal['n']} rows. The required-n table shows what a guarantee would need.",
          "- Budget routing needs the whole batch to take a quantile. Streaming needs a running quantile with a warmup window (docs/ROADMAP.md).",
          "- Lift intervals are wide at n of a few hundred. Treat differences between scorers whose intervals overlap as unresolved.", "",
          "## Reproduce", "", "```bash", "python -m routing.cli score --split calibration", "python -m routing.cli score --split test",
          "python -m routing.cli gates", "python -m routing.cli calibrate", "python -m routing.cli evaluate", "```", ""]
    return "\n".join(L)


def run() -> dict:
    a = assemble()
    m, cfg = a["df"], a["cfg"]
    cal = calibrate.load()
    g = json.loads((data.RESULTS / "gates_test.json").read_text())
    fl = gates.flags("test")
    lift = review_lift(m)
    ev = {"n_total": a["n_total"], "review_lift": lift, "cutoff_drift": cutoff_drift(m, cal), "cascade": cascade(m, cal),
          "cost_tiers": cost_tiers(m, lift), "risk_coverage": risk_coverage(m), "length_strata": length_strata(m),
          "conformal": conformal(m, cal), "prevalence": prevalence.run(_headline_rows(m)), "residual_curve": residual_curve(m)}
    ev["best_scorer"] = best_scorer(lift, fl)
    ev["best_free_scorer"] = best_free_scorer(lift, fl)
    data.FIGURES.mkdir(parents=True, exist_ok=True)
    data.REPORTS.mkdir(parents=True, exist_ok=True)
    fig_review_lift(lift, fl)
    fig_cascade(ev["cascade"], fl)
    fig_risk_coverage(ev["risk_coverage"], fl)
    prevalence.figure(ev["prevalence"], fl)
    fig_residual(ev["residual_curve"], fl)
    calls = cost.summary()
    (data.RESULTS / "evaluation.json").write_text(json.dumps(ev, indent=2, sort_keys=True, default=float))
    (data.REPORTS / "routing_report.md").write_text(render_report(ev, g, cal, fl, calls, cfg))
    return ev
