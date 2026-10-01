"""Phase 2: AUROC with 2000-resample bootstrap CIs, and AUROC power analysis.

Power uses the Hanley-McNeil (1982) AUROC variance with a one-sample z test of
H0: AUROC = 0.5 vs H1: AUROC = 0.65, two-sided alpha = 0.05, target power 0.80.
Run: python3 eval/audit/p2_power.py   -> out/p2_power.json, figures/p2_auroc_ci.png
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from common import FIG, auc_ci, col, f, load_mode, save  # noqa: E402

A1, ALPHA, TARGET = 0.65, 0.05, 0.80


def hm_var(a, n_pos, n_neg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)


def power(n_pos, n_neg, a1=A1, alpha=ALPHA):
    za = sps.norm.ppf(1 - alpha / 2)
    return float(sps.norm.cdf((a1 - 0.5 - za * math.sqrt(hm_var(0.5, n_pos, n_neg)))
                              / math.sqrt(hm_var(a1, n_pos, n_neg))))


def n_required(prev):
    for n in range(4, 200000):
        n_pos = round(prev * n)
        n_neg = n - n_pos
        if n_pos >= 1 and n_neg >= 1 and power(n_pos, n_neg) >= TARGET:
            return {"n": n, "pos": n_pos, "neg": n_neg}
    return None


def main():
    sets = {}
    for mode in ("api", "local"):
        rows = load_mode(mode)
        y, s = col(rows, "y").astype(int), col(rows, "hri_f")
        real = ~np.array([r["suppressed"] for r in rows])
        ya = col(rows, "ya").astype(int)
        sets[f"{mode}: as reported (SBERT label, all rows)"] = (y, s)
        sets[f"{mode}: real answers only (SBERT label)"] = (y[real], s[real])
        sets[f"{mode}: real answers only (audit label)"] = (ya[real], s[real])
        sets[f"{mode}: bound if every suppressed row was correct"] = (np.where(real, y, 0), s)

    out = {"assumptions": {"a1": A1, "alpha": ALPHA, "power": TARGET, "variance": "Hanley-McNeil"},
           "sets": {}, "required_n_balanced": n_required(0.5)}
    for name, (y, s) in sets.items():
        d = {"auroc": auc_ci(y, s)}
        if d["auroc"] and "bound" not in name:
            prev = d["auroc"]["pos"] / d["auroc"]["n"]
            d["power_at_actual_n"] = power(d["auroc"]["pos"], d["auroc"]["neg"])
            d["required_n_at_this_prevalence"] = n_required(prev)
        out["sets"][name] = d
        a = d["auroc"]
        extra = ""
        if "power_at_actual_n" in d:
            extra = (f"  power={d['power_at_actual_n']:.2f}  n_req={d['required_n_at_this_prevalence']}")
        print(f"{name:<52} n={a['n']:>3} pos={a['pos']:>3} neg={a['neg']:>2} "
              f"AUROC {f(a)} width={a['width']:.3f} contains0.5={a['contains_half']} "
              f"valid_boot={a['n_boot_valid']}{extra}")
    print("required n, balanced classes:", out["required_n_balanced"])
    save("p2_power", out)

    names = list(out["sets"])
    est = np.array([out["sets"][k]["auroc"]["est"] for k in names])
    lo = np.array([out["sets"][k]["auroc"]["lo"] for k in names])
    hi = np.array([out["sets"][k]["auroc"]["hi"] for k in names])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yy = np.arange(len(names))
    ax.errorbar(est, yy, xerr=[est - lo, hi - est], fmt="o", color="#2563eb", capsize=3)
    ax.axvline(0.5, ls="--", color="grey")
    ax.set_yticks(yy, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUROC of HRI (95% bootstrap CI, 2000 resamples)")
    fig.tight_layout()
    fig.savefig(FIG / "p2_auroc_ci.png", dpi=130)


if __name__ == "__main__":
    main()
