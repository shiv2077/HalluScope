"""Phase 4: cost of the intervention rule (answer iff HRI < threshold).

Coverage = fraction answered. Selective risk = error rate among answered.
AURC = mean selective risk over all coverage levels (rows sorted by HRI).
Run: python3 eval/audit/p4_risk_coverage.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIG, boot, col, f, load_mode, save, wilson  # noqa: E402

T = 0.30  # config/thresholds.json risk_high (the deployed intervention trigger)


def rc(y, s):
    order = np.argsort(s, kind="stable")
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(y[order]) / k


def aurc(y, s):
    return float(np.mean(rc(y, s)[1]))


def summarize(y, s):
    flag, ans = s >= T, s < T
    return {
        "n": len(y),
        "aurc": boot(aurc, y, s),
        "oracle_aurc": boot(lambda a: aurc(a, a), y),
        "excess_aurc": boot(lambda a, b: aurc(a, b) - aurc(a, a), y, s),
        "risk_answer_everything": wilson(int(y.sum()), len(y)),
        "coverage_at_T": wilson(int(ans.sum()), len(y)),
        "selective_risk_at_T": wilson(int(y[ans].sum()), int(ans.sum())),
        "risk_reduction_at_T": boot(lambda a, b: float(a.mean() - a[b < T].mean())
                                    if (b < T).any() else None, y, s),
        "wrong_withheld_at_T": int((flag & (y == 1)).sum()),
        "correct_suppressed_at_T": int((flag & (y == 0)).sum()),
        "net_wrong_withheld_minus_correct_suppressed": boot(
            lambda a, b: float(((b >= T) & (a == 1)).sum() - ((b >= T) & (a == 0)).sum()), y, s),
    }


def main():
    res, curves = {}, {}
    for mode in ("api", "local"):
        rows = load_mode(mode)
        y, s = col(rows, "y").astype(int), col(rows, "hri_f")
        real = ~np.array([r["suppressed"] for r in rows])
        ya = col(rows, "ya").astype(int)
        sets = {"SBERT, all rows (as reported)": (y, s),
                "SBERT, real answers": (y[real], s[real]),
                "audit, real answers": (ya[real], s[real])}
        res[mode] = {k: summarize(*v) for k, v in sets.items()}
        res[mode]["suppressed_rows_flagged_at_T_label_unknown"] = int(((s >= T) & ~real).sum())
        res[mode]["risk_answer_everything_bounds_all_rows"] = [
            float(y[real].sum() / len(y)), float((y[real].sum() + (~real).sum()) / len(y))]
        curves[mode] = sets
        print(f"\n== {mode} (threshold {T}); suppressed rows flagged with unknown label: "
              f"{res[mode]['suppressed_rows_flagged_at_T_label_unknown']}; "
              f"answer-everything risk bounds {res[mode]['risk_answer_everything_bounds_all_rows']}")
        for k in sets:
            d = res[mode][k]
            print(f"  {k:<32} n={d['n']:>3} AURC {f(d['aurc'])} oracle {f(d['oracle_aurc'])} "
                  f"excess {f(d['excess_aurc'])}")
            print(f"  {'':<32} coverage@T {f(d['coverage_at_T'])} sel.risk@T {f(d['selective_risk_at_T'])} "
                  f"risk(all) {f(d['risk_answer_everything'])} reduction {f(d['risk_reduction_at_T'])}")
            print(f"  {'':<32} wrong withheld {d['wrong_withheld_at_T']} correct suppressed "
                  f"{d['correct_suppressed_at_T']} net {f(d['net_wrong_withheld_minus_correct_suppressed'], 1)}")
    save("p4_risk_coverage", res)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, mode in zip(axes, ("api", "local")):
        for (k, (y, s)), colr in zip(curves[mode].items(), ("#dc2626", "#2563eb", "#16a34a")):
            cov, risk = rc(y, s)
            ax.plot(cov, risk, color=colr, label=f"{k} (AURC {res[mode][k]['aurc']['est']:.3f})")
            ax.axhline(y.mean(), color=colr, ls=":", lw=1)
            c = (s < T).mean()
            ax.scatter([c], [y[s < T].mean() if (s < T).any() else np.nan], color=colr,
                       marker="x", s=70, zorder=4)
        ax.set_xlabel("coverage (fraction answered, rows sorted by HRI)")
        ax.set_ylabel("selective risk (error rate among answered)")
        ax.set_title(f"{mode}: x = HRI threshold {T}; dotted = answer everything", fontsize=9)
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "p4_risk_coverage.png", dpi=130)


if __name__ == "__main__":
    main()
