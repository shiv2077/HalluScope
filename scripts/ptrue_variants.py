"""P(True) scoring variants on the routing test split, plus the length-control tie count.

All variants are recomputed from the stored ptrue generations, so the comparison regenerates on CPU
without git history or API calls:
  p_yes_6dp: 1 - P(Yes) stored at 6 decimals
  p_yes_full: 1 - P(Yes) at full precision
  logodds: routing.scorers.p_true, No-over-Yes log-odds plus a mean_logprob tie term (ADR-019)
Also reports a sensitivity check with the rows where "No" was absent from the top 5 dropped, and how
many rows of the length control are tied at each budget cutoff (ADR-020).
Writes results/ptrue_variants.json and results/ptrue_variants.md. Run: python3 scripts/ptrue_variants.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing import data, evaluate, scorers, stats  # noqa: E402

BUDGETS = (0.10, 0.20, 0.30)


def p_yes_risk(first_top: list) -> float:
    return 1.0 - sum(math.exp(t["logprob"]) for t in first_top if t["token"].strip().lower() == "yes")


def straddling_ties(s: np.ndarray, budget: float) -> dict:
    """Rows whose flag under the budget policy is decided by tie order rather than by the score."""
    k = math.ceil(budget * len(s))
    cut = np.sort(s)[::-1][k - 1]
    group = int((s == cut).sum())
    above = int((s > cut).sum())
    taken = k - above
    return {"k": k, "tied_group_at_cutoff": group, "flagged_from_group": taken,
            "rows_decided_by_order": group if 0 < taken < group else 0}


def main() -> None:
    a = evaluate.assemble()
    m = a["df"]
    g = data.read_generations("test").set_index(["qid", "role"])
    tops = {q: json.loads(g.loc[(q, "ptrue"), "token_logprobs"]) for q in m["qid"]}
    m["p_yes_full"] = [p_yes_risk(tops[q]) for q in m["qid"]]
    m["p_yes_6dp"] = m["p_yes_full"].round(6)
    m["logodds"] = [scorers.p_true(tops[q], ml) for q, ml in zip(m["qid"], m["mean_logprob"])]
    m["no_absent"] = [scorers.p_true_logodds(tops[q])[2] for q in m["qid"]]
    h = m[m["label_cheap"].isin(["CORRECT", "INCORRECT"])].reset_index(drop=True)
    y = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    out = {"n": int(len(h)), "positives": int(y.sum()), "variants": {}}
    for name in ("p_yes_6dp", "p_yes_full", "logodds"):
        s = h[name].to_numpy(float)
        v = {"auroc": stats.auc_ci(y, s), "distinct_values": int(len(np.unique(s))), "budgets": {}}
        for b in BUDGETS:
            v["budgets"][str(b)] = {"lift": stats.boot(lambda yy, ss, b=b: evaluate._lift(yy, ss, b), y, s),
                                    "catch_rate": stats.boot(lambda yy, ss, b=b: evaluate._catch(yy, ss, b), y, s),
                                    "ties": straddling_ties(s, b)}
        out["variants"][name] = v
    out["auroc_diff_logodds_minus_p_yes_full"] = stats.delong(y, h["logodds"].to_numpy(float), h["p_yes_full"].to_numpy(float))
    out["auroc_diff_logodds_minus_p_yes_6dp"] = stats.delong(y, h["logodds"].to_numpy(float), h["p_yes_6dp"].to_numpy(float))
    keep = ~h["no_absent"].to_numpy(bool)
    out["sensitivity_no_absent_dropped"] = {
        "n_dropped": int((~keep).sum()), "n": int(keep.sum()),
        "auroc": stats.auc_ci(y[keep], h["logodds"].to_numpy(float)[keep]),
        "lift_0.2": stats.boot(lambda yy, ss: evaluate._lift(yy, ss, 0.2), y[keep], h["logodds"].to_numpy(float)[keep])}
    both = m[m["label_cheap"].isin(["CORRECT", "INCORRECT"]) & m["label_strong"].isin(["CORRECT", "INCORRECT"])]
    yc = (both["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    ys = (both["label_strong"] == "INCORRECT").astype(int).to_numpy()
    cc, cs = both["cost_cheap"].to_numpy(float), both["cost_strong"].to_numpy(float)
    extra = np.array([e["p_true"] for e in both["extra_cost"]], float)
    out["cascade"] = {}
    for name in ("p_yes_6dp", "p_yes_full", "logodds"):
        s = both[name].to_numpy(float)
        out["cascade"][name] = {str(b): evaluate._cascade_point(evaluate._flags(s, b), yc, ys, cc, cs, extra, None, "budget")
                                for b in BUDGETS}
    L = h["length"].to_numpy(float)
    out["length_ties"] = {"distinct_values": int(len(np.unique(L))),
                          "budgets": {str(b): straddling_ties(L, b) for b in BUDGETS}}
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    (data.RESULTS / "ptrue_variants.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    (data.RESULTS / "ptrue_variants.md").write_text(render(out))
    print(f"wrote {data.RESULTS / 'ptrue_variants.json'} and ptrue_variants.md")


def _pct(d):
    return "n/a" if d is None else f"{100 * d['est']:.1f}% [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"


def render(o: dict) -> str:
    names = ("p_yes_6dp", "p_yes_full", "logodds")
    V = [o["variants"][n] for n in names]
    d = o["auroc_diff_logodds_minus_p_yes_full"]
    d2 = o["auroc_diff_logodds_minus_p_yes_6dp"]
    L = ["# P(True) scoring variants", "",
         f"Routing test split, {o['n']} rows with a consensus label ({o['positives']} INCORRECT). Brackets are 95 percent "
         "bootstrap intervals (2000 resamples, seed 42). Budget policy throughout. Regenerate with `python3 scripts/compare_ptrue.py`.", "",
         "Columns: 1 - P(Yes) stored at 6 decimals; the same at full precision; and No-over-Yes log-odds with the mean_logprob tie term, "
         "the scorer routing uses.", "",
         "| | 1 - P(Yes), 6 decimals | 1 - P(Yes), full precision | log-odds (used) |", "|---|---|---|---|",
         "| distinct score values | " + " | ".join(str(v["distinct_values"]) for v in V) + " |",
         "| AUROC | " + " | ".join(stats.f(v["auroc"]) for v in V) + " |"]
    for b in BUDGETS:
        L.append(f"| lift at {b:.2f} | " + " | ".join(stats.f(v["budgets"][str(b)]["lift"], 2) for v in V) + " |")
        L.append(f"| rows at {b:.2f} in a tie straddling the cutoff | " + " | ".join(
            f"{v['budgets'][str(b)]['ties']['rows_decided_by_order']} (tied group {v['budgets'][str(b)]['ties']['tied_group_at_cutoff']})" for v in V) + " |")
    for b in BUDGETS:
        L.append(f"| cascade error at {b:.2f} | " + " | ".join(_pct(o["cascade"][n][str(b)]["error"]) for n in names) + " |")
    s = o["sensitivity_no_absent_dropped"]
    lt = o["length_ties"]
    L += ["", f"AUROC difference, log-odds minus 1 - P(Yes) at full precision: {stats.f(d['diff'])}, DeLong p = {d['p_delong']:.3f}. "
          f"Log-odds minus 1 - P(Yes) at 6 decimals: {stats.f(d2['diff'])}, DeLong p = {d2['p_delong']:.3f}.", "",
          f"Sensitivity: dropping the {s['n_dropped']} rows where \"No\" was absent from the top 5 and imputed, the log-odds scorer has "
          f"AUROC {stats.f(s['auroc'])} and lift at 0.20 of {stats.f(s['lift_0.2'], 2)} on {s['n']} rows.", "",
          "## Length-control ties (ADR-020)", "",
          f"The `length` control takes {lt['distinct_values']} distinct values on these rows, so the budget policy breaks its ties "
          "by qid. Rows in the tied group at each cutoff:", "",
          "| budget | rows flagged | tied group at cutoff | rows ordered by qid |", "|---:|---:|---:|---:|"]
    for b in BUDGETS:
        t = lt["budgets"][str(b)]
        L.append(f"| {b:.2f} | {t['k']} | {t['tied_group_at_cutoff']} | {t['rows_decided_by_order']} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
