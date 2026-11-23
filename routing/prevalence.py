"""Prevalence sensitivity: what review lift and flagged-slice precision look like at production error rates.

Keep every CORRECT test row, subsample INCORRECT rows without replacement to a target prevalence, repeat
300 times (seeds 0..299), and summarize each metric by its median and 2.5 / 97.5 percentiles across
subsamples. These are subsample intervals, not row bootstraps.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data  # noqa: E402
from .policy import flag_by_budget  # noqa: E402

TARGETS = (0.05, 0.10, 0.15, 0.25)
N_REP = 300
BUDGET = 0.2
COLORS = {"mean_logprob": "#1f77b4", "min_k_logprob": "#ff7f0e", "length": "#7f7f7f",
          "length_residual": "#2ca02c", "consistency_k3": "#d62728", "p_true": "#9467bd"}


def _metrics(y: np.ndarray, s: np.ndarray) -> dict:
    fl = flag_by_budget(pd.Series(s), BUDGET).to_numpy()
    n_err = y.sum()
    return {"catch": float(y[fl].sum() / n_err) if n_err else np.nan,
            "lift": float((y[fl].sum() / n_err) / (fl.sum() / len(fl))) if n_err and fl.sum() else np.nan,
            "precision": float(y[fl].mean()) if fl.sum() else np.nan,
            "residual": float(y[~fl].mean()) if (~fl).sum() else np.nan}


def _summ(vals: list) -> dict:
    a = np.array(vals, float)
    a = a[np.isfinite(a)]
    return {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)), "n_valid": int(len(a))}


def run(h: pd.DataFrame) -> dict:
    """h: headline rows with label_cheap in CORRECT/INCORRECT and scorer columns."""
    y_all = (h["label_cheap"] == "INCORRECT").astype(int).to_numpy()
    correct = np.where(y_all == 0)[0]
    wrong = np.where(y_all == 1)[0]
    observed = float(y_all.mean())
    out = {"observed_prevalence": observed, "n_correct": int(len(correct)), "n_incorrect": int(len(wrong)),
           "n_rep": N_REP, "budget": BUDGET, "targets": {}}
    S = {sc: h[sc].astype(float).to_numpy() for sc in data.SCORERS}
    for t in TARGETS:
        k = int(round(t * len(correct) / (1 - t)))
        if k < 1 or k > len(wrong):
            out["targets"][str(t)] = {"skipped": f"needs {k} INCORRECT rows, have {len(wrong)}"}
            continue
        acc = {sc: {m: [] for m in ("catch", "lift", "precision", "residual")} for sc in data.SCORERS}
        for seed in range(N_REP):
            rng = np.random.RandomState(seed)
            idx = np.concatenate([correct, rng.choice(wrong, size=k, replace=False)])
            idx.sort()
            y = y_all[idx]
            for sc in data.SCORERS:
                s = S[sc][idx]
                ok = np.isfinite(s)
                for m, v in _metrics(y[ok], s[ok]).items():
                    acc[sc][m].append(v)
        out["targets"][str(t)] = {"k_incorrect": k, "n": int(len(correct) + k),
                                  "scorers": {sc: {m: _summ(v) for m, v in ms.items()} for sc, ms in acc.items()}}
    # observed point, single evaluation on the full set (no subsampling)
    out["observed"] = {sc: _metrics(y_all[np.isfinite(S[sc])], S[sc][np.isfinite(S[sc])]) for sc in data.SCORERS}
    return out


def figure(pv: dict, fl: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ts = [float(t) for t in pv["targets"] if "skipped" not in pv["targets"][t]]
    for ax, metric, ylabel in ((axes[0], "lift", "lift at 0.20 budget"), (axes[1], "precision", "precision of flagged slice")):
        for sc in data.SCORERS:
            med = [pv["targets"][str(t)]["scorers"][sc][metric]["median"] for t in ts]
            lo = [pv["targets"][str(t)]["scorers"][sc][metric]["lo"] for t in ts]
            hi = [pv["targets"][str(t)]["scorers"][sc][metric]["hi"] for t in ts]
            label = "length (control)" if sc == "length" else sc + (" (not distinguishable from length)" if fl.get(sc) else "")
            ax.plot(ts, med, "--" if sc == "length" else "-", marker="o", ms=4, color=COLORS[sc], label=label)
            ax.fill_between(ts, lo, hi, color=COLORS[sc], alpha=0.12)
            ax.plot([pv["observed_prevalence"]], [pv["observed"][sc][metric]], "D", color=COLORS[sc], ms=6)
        ax.axvline(pv["observed_prevalence"], color="black", lw=0.8, ls=":")
        ax.set_xlabel("error prevalence (diamonds: observed test set)")
        ax.set_ylabel(ylabel)
        if metric == "lift":
            ax.axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0].legend(fontsize=6)
    fig.suptitle("Prevalence sensitivity, budget policy at 0.20 (bands: 2.5 to 97.5 percentiles over 300 subsamples)")
    fig.tight_layout()
    fig.savefig(data.FIGURES / "prevalence_sensitivity.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


def _f(d, pct=False):
    if pct:
        return f"{100 * d['median']:.1f}% [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    return f"{d['median']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def render(pv: dict, best: str, fl: dict) -> list[str]:
    L = ["", "### Prevalence sensitivity", "",
         f"Every CORRECT test row is kept ({pv['n_correct']}) and INCORRECT rows are subsampled without replacement to the target "
         f"prevalence, {pv['n_rep']} times (seeds 0 to {pv['n_rep'] - 1}). Cells are the median and the 2.5 and 97.5 percentiles "
         "across subsamples. These are subsample intervals, not row bootstraps. Budget policy at 0.20 inside each subsample. "
         f"The observed test prevalence is {100 * pv['observed_prevalence']:.1f}%.", "",
         "![prevalence sensitivity](../figures/prevalence_sensitivity.png)", "",
         "| scorer | prevalence | n | catch rate | lift | precision | residual error |", "|---|---:|---:|---|---|---|---|"]
    for sc in data.SCORERS:
        for t, blk in pv["targets"].items():
            if "skipped" in blk:
                L.append(f"| {sc} | {float(t):.2f} | | skipped: {blk['skipped']} | | | |")
                continue
            r = blk["scorers"][sc]
            star = " *" if fl.get(sc) and sc != "length" else ""
            L.append(f"| {sc}{star} | {float(t):.2f} | {blk['n']} | {_f(r['catch'], True)} | {_f(r['lift'])} | "
                     f"{_f(r['precision'], True)} | {_f(r['residual'], True)} |")
    lo_t, hi_t = min(pv["targets"], key=float), max(pv["targets"], key=float)
    b_lo, b_hi = pv["targets"][lo_t]["scorers"][best]["lift"], pv["targets"][hi_t]["scorers"][best]["lift"]
    holds = b_lo["lo"] > 1.0
    p10 = pv["targets"]["0.1"]["scorers"][best]["precision"]
    L += ["", f"Lift for `{best}` is {_f(b_hi)} at prevalence {float(hi_t):.2f} and {_f(b_lo)} at {float(lo_t):.2f}, so it "
          f"{'holds above 1 across the whole range' if holds else 'is not reliably above 1 at the lowest prevalence'} as prevalence falls. "
          f"At 10 percent prevalence a reviewer working the flagged slice from `{best}` would see precision {_f(p10, True)}: "
          f"roughly {int(round(1 / p10['median']))} rows read per error found, against {int(round(1 / 0.10))} at random."]
    return L
