"""Evaluation on the test split only, using cutoffs from results/calibration.json. Never recomputes a cutoff.

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

from . import calibrate, cost, data, gates, stats  # noqa: E402

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
    return {"df": m, "n_total": len(m), "cfg": cfg}


def _headline_rows(m: pd.DataFrame) -> pd.DataFrame:
    return m[m["label_cheap"].isin(["CORRECT", "INCORRECT"])].copy()


# ------------------------------------------------------------------ 1. review lift
def _lift_stats(y, flagged):
    n_err = y.sum()
    if n_err == 0 or flagged.sum() == 0:
        return None
    return float(y[flagged].sum() / n_err)


def review_lift(m: pd.DataFrame, cal: dict) -> dict:
    out = {}
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    out["_n"] = int(len(h))
    out["_error_rate"] = stats.wilson(int(y_all.sum()), len(h))
    for sc in data.SCORERS:
        s = h[sc].astype(float).to_numpy()
        ok = np.isfinite(s)
        y = y_all[ok]
        s = s[ok]
        out[sc] = {}
        for b in BUDGETS:
            cut = cal["budgets"][sc][str(b)]
            fl = s >= cut
            realized = stats.wilson(int(fl.sum()), len(fl))
            catch = stats.boot(lambda yy, ff: _lift_stats(yy, ff.astype(bool)), y, fl)
            lift = stats.boot(lambda yy, ff: (None if ff.sum() == 0 or yy.sum() == 0 else
                                              (yy[ff.astype(bool)].sum() / yy.sum()) / (ff.sum() / len(ff))), y, fl)
            resid = stats.boot(lambda yy, ff: None if (~ff.astype(bool)).sum() == 0 else float(yy[~ff.astype(bool)].mean()), y, fl)
            out[sc][str(b)] = {"cutoff": cut, "flagged_fraction": realized, "catch_rate": catch,
                               "random_catch_rate": realized, "lift": lift, "residual_error_rate": resid}
    return out


def fig_review_lift(lift: dict, fl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sc in data.SCORERS:
        xs = [lift[sc][str(b)]["flagged_fraction"]["est"] for b in BUDGETS]
        ys = [lift[sc][str(b)]["catch_rate"] for b in BUDGETS]
        if any(v is None for v in ys):
            continue
        est = [v["est"] for v in ys]
        lo = [v["lo"] for v in ys]
        hi = [v["hi"] for v in ys]
        style = "--" if sc == "length" else "-"
        label = sc + (" (not distinguishable from length)" if fl.get(sc) and sc != "length" else "")
        label = "length (control)" if sc == "length" else label
        marker = "x" if fl.get(sc) else "o"
        ax.plot(xs, est, style, marker=marker, color=COLORS[sc], label=label)
        ax.fill_between(xs, lo, hi, color=COLORS[sc], alpha=0.12)
    ax.plot([0, 0.35], [0, 0.35], ":", color="black", label="random")
    ax.set_xlabel("review budget (fraction of rows flagged)")
    ax.set_ylabel("catch rate (fraction of errors inside flagged slice)")
    ax.set_title("Review lift on the test split (bands: 95% bootstrap)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(data.FIGURES / "review_lift.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------ 2. cascade frontier
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
        cuts = {f"budget_{b}": cal["budgets"][sc][str(b)] for b in BUDGETS}
        for a in ALPHAS:
            r = cal["risk"][sc][str(a)]
            if r.get("threshold") is not None:
                cuts[f"risk_{a}"] = r["threshold"]
        for name, cut in cuts.items():
            esc = np.isfinite(s) & (s >= cut) if name.startswith("budget") else np.isfinite(s) & (s > cut)
            y = np.where(esc, ys, yc)
            c = cc + extra + np.where(esc, cs, 0.0)
            fix_n = int(((yc == 1) & esc).sum())
            fix_k = int(((yc == 1) & esc & (ys == 0)).sum())
            brk_n = int(((yc == 0) & esc).sum())
            brk_k = int(((yc == 0) & esc & (ys == 1)).sum())
            points[name] = {"cutoff": cut, "escalated_fraction": stats.wilson(int(esc.sum()), len(esc)),
                            "error": stats.boot(lambda yy: float(yy.mean()), y),
                            "cost_per_1000": stats.boot(lambda x: float(x.mean()) * 1000, c),
                            "fix_rate": stats.wilson(fix_k, fix_n), "break_rate": stats.wilson(brk_k, brk_n)}
        out["scorers"][sc] = points
    return out


def fig_cascade(cas: dict, fl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sc, pts in cas["scorers"].items():
        bp = [pts[f"budget_{b}"] for b in BUDGETS]
        xs = [p["cost_per_1000"]["est"] for p in bp]
        ys = [p["error"]["est"] for p in bp]
        ye = [[p["error"]["est"] - p["error"]["lo"] for p in bp], [p["error"]["hi"] - p["error"]["est"] for p in bp]]
        label = "length (control)" if sc == "length" else sc + (" (not distinguishable from length)" if fl.get(sc) else "")
        ax.errorbar(xs, ys, yerr=ye, fmt=("x--" if fl.get(sc) else "o-"), color=COLORS[sc], label=label, capsize=2, alpha=0.9)
    for name, mk in (("all_cheap", "s"), ("all_strong", "D")):
        p = cas[name]
        ax.plot(p["cost_per_1000"]["est"], p["error"]["est"], mk, ms=11, color="black", label=name.replace("_", " "))
    ax.set_xlabel("cost per 1000 queries (USD, config/pricing.json)")
    ax.set_ylabel("error rate after routing")
    ax.set_title("Cascade frontier on the test split (budget cutoffs from calibration)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(data.FIGURES / "cascade_frontier.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------ 3. risk-coverage
def _aurc(y, s):
    order = np.argsort(s)  # accept lowest risk first
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
        y = y_all[ok]
        s = s[ok]
        oracle = _aurc(y, y.astype(float))  # errors sorted last
        order = np.argsort(s)
        ys = y[order]
        curve_risk = (np.cumsum(ys) / np.arange(1, len(ys) + 1)).tolist()
        curve_cov = (np.arange(1, len(ys) + 1) / len(ys)).tolist()
        out[sc] = {"aurc": stats.boot(_aurc, y, s), "oracle_aurc": oracle,
                   "curve": {"coverage": curve_cov[::max(1, len(ys) // 60)], "risk": curve_risk[::max(1, len(ys) // 60)]}}
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


# ------------------------------------------------------------------ 4. length stratification
def length_strata(m: pd.DataFrame, cal: dict) -> dict:
    h = _headline_rows(m)
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    L = h["length"].to_numpy(float)
    edges = np.quantile(L, [1 / 3, 2 / 3])
    tert = np.digitize(L, edges)
    out = {"edges": edges.tolist(), "n_per_tertile": [int((tert == t).sum()) for t in range(3)]}
    for sc in data.SCORERS:
        s = h[sc].astype(float).to_numpy()
        cut = cal["budgets"][sc]["0.2"]
        out[sc] = []
        for t in range(3):
            sel = (tert == t) & np.isfinite(s)
            y = y_all[sel]
            fl = (s[sel] >= cut)
            lift = stats.boot(lambda yy, ff: (None if ff.sum() == 0 or yy.sum() == 0 else
                                              (yy[ff.astype(bool)].sum() / yy.sum()) / (ff.sum() / len(ff))), y, fl)
            out[sc].append({"tertile": t, "n": int(sel.sum()), "errors": int(y.sum()),
                            "flagged": int(fl.sum()), "lift": lift})
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
    pool = unflagged or cands
    return max(pool)[1]


def render_report(ev: dict, g: dict, cal: dict, fl: dict, calls: dict, cfg: dict) -> str:
    lift, cas, rc, ls, cf = ev["review_lift"], ev["cascade"], ev["risk_coverage"], ev["length_strata"], ev["conformal"]
    best = ev["best_scorer"]
    b20 = lift[best]["0.2"]
    # best cascade operating point: lowest error among budget points of the best scorer
    pts = cas["scorers"][best]
    bp_name = min((k for k in pts if k.startswith("budget")), key=lambda k: pts[k]["error"]["est"])
    bp = pts[bp_name]
    tag = " (not distinguishable from length)" if fl.get(best) else ""
    L = ["# Routing report", "",
         f"Prices from config/pricing.json (priced_on {calls['priced_on']}). Brackets are 95 percent bootstrap "
         "intervals (2000 resamples, seed 42) or Wilson intervals for proportions. Raw counts carry no interval.", "",
         "## Summary", "",
         f"On {lift['_n']} test rows with a consensus label, the cheap model ({cfg['cheap']}) has an error rate of {_pct(lift['_error_rate'])}. "
         f"The best scorer by lift at a 20 percent review budget is `{best}`{tag}, catching {_pct(b20['catch_rate'])} of errors "
         f"against {_pct(b20['random_catch_rate'])} for random, a lift of {stats.f(b20['lift'], 2)}. "
         f"Residual error among unreviewed rows at that budget is {_pct(b20['residual_error_rate'])}. "
         f"The cascade at the `{best}` {bp_name.replace('_', ' ')} cutoff has error {_pct(bp['error'])} at {_usd(bp['cost_per_1000'])} per thousand queries, "
         f"against all-cheap {_pct(cas['all_cheap']['error'])} at {_usd(cas['all_cheap']['cost_per_1000'])} and all-strong {_pct(cas['all_strong']['error'])} at {_usd(cas['all_strong']['cost_per_1000'])}. "
         f"Every cutoff was computed on the {cal['n']}-row calibration split and applied once to test.", "",
         "## Data and labels", "",
         f"- Calibration: {cal['n']} rows with a consensus label ({cal['positives']} INCORRECT), {cal['n_ambiguous_excluded']} AMBIGUOUS excluded. "
         f"Scores file hash {cal['calibration_hash']}, labels hash {cal['labels_hash']}.",
         f"- Test: {ev['n_total']} questions, {lift['_n']} with a consensus cheap-model label, {ev['n_total'] - lift['_n']} AMBIGUOUS excluded. "
         f"Cascade uses the {cas['_n']} rows where both models have a consensus label ({cas['_n_excluded']} excluded).",
         f"- Judges: `{cfg['judge_a']}` and `{cfg['judge_b']}`, blind to every score, consensus rule in routing/data.py. Agreement and kappa in results/phase1_notes.md.", "",
         "## Validity gates", "", f"See results/gates_test.md. Template rows: {g['template_rows']}. Split disjoint: {g['disjoint']}. "
         f"Power at actual n: {g['power_at_actual_n']:.2f}." if g["power_at_actual_n"] is not None else "See results/gates_test.md.", "",
         "| scorer | AUROC | vs length p | flag |", "|---|---|---|---|"]
    for sc, r in g["scorers"].items():
        vl = r.get("vs_length")
        L.append(f"| {sc} | {stats.f(r['auroc'])} | {vl['p_delong']:.3f} | {'control' if sc == 'length' else ('' if r['beats_length'] else 'not distinguishable from length')} |" if vl
                 else f"| {sc} | {stats.f(r['auroc'])} | n/a | control |")
    L += ["", "## Review lift", "", "![review lift](../figures/review_lift.png)", "",
          "| scorer | budget | flagged | catch rate | random | lift | residual error |", "|---|---:|---|---|---|---|---|"]
    for sc in data.SCORERS:
        for b in BUDGETS:
            r = lift[sc][str(b)]
            L.append(f"| {sc}{' *' if fl.get(sc) and sc != 'length' else ''} | {b:.2f} | {_pct(r['flagged_fraction'])} | {_pct(r['catch_rate'])} | "
                     f"{_pct(r['random_catch_rate'])} | {stats.f(r['lift'], 2)} | {_pct(r['residual_error_rate'])} |")
    L += ["", "`*` marks a scorer not distinguishable from length. `length` is the control. Residual error is the error rate among "
          "unflagged rows, assuming review fixes what it finds.", "",
          "## Cascade frontier", "", "![cascade frontier](../figures/cascade_frontier.png)", "",
          f"Endpoints: all cheap error {_pct(cas['all_cheap']['error'])} at {_usd(cas['all_cheap']['cost_per_1000'])} per 1000; "
          f"all strong error {_pct(cas['all_strong']['error'])} at {_usd(cas['all_strong']['cost_per_1000'])} per 1000. "
          f"The strong model's error rate is the ceiling of any cascade.", "",
          "| scorer | cutoff | escalated | error | cost per 1000 | fix rate | break rate |", "|---|---|---|---|---|---|---|"]
    for sc, pts in cas["scorers"].items():
        for name, p in pts.items():
            L.append(f"| {sc}{' *' if fl.get(sc) and sc != 'length' else ''} | {name} | {_pct(p['escalated_fraction'])} | {_pct(p['error'])} | "
                     f"{_usd(p['cost_per_1000'])} | {_pct(p['fix_rate'])} | {_pct(p['break_rate'])} |")
    L += ["", "Fix rate: among escalated rows the cheap model got wrong, the fraction the strong model got right. "
          "Break rate: among escalated rows the cheap model got right, the fraction the strong model got wrong.", "",
          "## Risk-coverage", "", "![risk coverage](../figures/risk_coverage.png)", "",
          "| scorer | AURC | oracle AURC |", "|---|---|---:|"]
    for sc in data.SCORERS:
        L.append(f"| {sc} | {stats.f(rc[sc]['aurc'])} | {rc[sc]['oracle_aurc']:.3f} |")
    L += ["", "## Length", "", f"Length tertile edges (output tokens): {ls['edges'][0]:.0f}, {ls['edges'][1]:.0f}. "
          f"Rows per tertile: {ls['n_per_tertile']}. Lift at the 0.20 budget cutoff within each tertile.", "",
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
    L += ["", "## Limitations", "",
          "- Short-answer QA (HaluEval) only. Long multi-claim outputs need claim-level scoring (ROADMAP.md).",
          "- OpenAI models only for generation. Logprob scorers need a provider that returns logprobs; consistency and P(True) do not.",
          "- Labels are from two LLM judges, not humans. AMBIGUOUS rows are excluded and counted above.",
          f"- Conformal thresholds are calibrated on {cal['n']} rows. The required-n table shows what a guarantee would need.",
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
    ev = {"n_total": a["n_total"], "review_lift": review_lift(m, cal), "cascade": cascade(m, cal),
          "risk_coverage": risk_coverage(m), "length_strata": length_strata(m, cal), "conformal": conformal(m, cal)}
    ev["best_scorer"] = best_scorer(ev["review_lift"], fl)
    data.FIGURES.mkdir(parents=True, exist_ok=True)
    data.REPORTS.mkdir(parents=True, exist_ok=True)
    fig_review_lift(ev["review_lift"], fl)
    fig_cascade(ev["cascade"], fl)
    fig_risk_coverage(ev["risk_coverage"], fl)
    calls = cost.summary()
    (data.RESULTS / "evaluation.json").write_text(json.dumps(ev, indent=2, sort_keys=True, default=float))
    (data.REPORTS / "routing_report.md").write_text(render_report(ev, g, cal, fl, calls, cfg))
    return ev
