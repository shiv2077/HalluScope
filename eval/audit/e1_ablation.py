"""E1: does the HRI ensemble beat its own single features?

Data (committed; no answers regenerated):
  * eval/results_api.csv   reported API run. Only hri and mean_logprob were logged, so
    0.4*var + 0.3*decay is recovered from hri by inverting the heuristic sigmoid.
  * eval/results_local.csv reported local run (as-run formula: |E|/35 energy, 2x RCD).
  * eval/gold/gold_results_api.csv  80 gpt-4o-mini answers from the current code with
    every HRI feature logged; labelled by SBERT at 0.35 and by the audit judgements.
Run: python3 eval/audit/e1_ablation.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import (FIG, auc_ci, delong_diff, f, load_gold, load_mode, raw_from_hri,  # noqa: E402
                    save, spearman_ci)

HRI = "HRI (full)"
MEAN = "|mean logprob| (length-normalized)"
TOK = "answer tokens"


def num(v):
    return float("nan") if v is None else float(v)


def en_as_run(r):
    return min(abs(float(r["mean_semantic_energy"])) / 35.0, 1.0)


def rcd_inv_as_run(r):
    return 1.0 - min(float(r["mean_rcd"]) * 2.0, 1.0)


def run(name, rows, feats, label_sets):
    out = {}
    for lname, mask in label_sets:
        sub = [r for r, m in zip(rows, mask) if m]
        yy = np.array([r["y"] if lname.startswith("SBERT") else r["ya"] for r in sub])
        scores = {k: np.array([fn(r) for r in sub]) for k, fn in feats.items()}
        scores = {k: v for k, v in scores.items() if np.isfinite(v).all()}
        aucs = {k: auc_ci(yy, v) for k, v in scores.items()}
        singles = [k for k in aucs if k not in (HRI, TOK) and aucs[k]]
        best = max(singles, key=lambda k: aucs[k]["est"])
        dl = delong_diff(yy, scores[HRI], scores[best])
        out[lname] = {"aucs": aucs, "best_single": best, "hri_vs_best": dl}
        print(f"\n[{name} | {lname}] n={len(yy)} pos={int(yy.sum())} neg={int(len(yy) - yy.sum())}")
        for k, d in aucs.items():
            print(f"  {k:<42} {f(d)}")
        print(f"  DeLong HRI vs {best}: p={dl['p_delong']:.3f}  AUROC diff {f(dl['diff'])}")
    return out


def corr_matrix(rows, feats):
    rows = [r for r in rows if r["tokens"] is not None]
    cols = {k: np.array([fn(r) for r in rows]) for k, fn in feats.items()}
    cols = {k: v for k, v in cols.items() if np.isfinite(v).all()}
    return {"n": len(rows),
            "matrix": {a: {b: spearman_ci(cols[a], cols[b]) for b in cols} for a in cols}}


def main():
    api = load_mode("api")
    local = load_mode("local")
    gold = load_gold()
    everything = [True] * 100
    real_api = [not r["suppressed"] for r in api]
    real_local = [not r["suppressed"] for r in local]

    base = {HRI: lambda r: r["hri_f"], MEAN: lambda r: abs(float(r["mean_logprob"]))}
    api_feats = {
        **base,
        "0.4*var + 0.3*decay (derived from hri)":
            lambda r: raw_from_hri(r["hri_f"]) - 0.3 * abs(float(r["mean_logprob"])),
        TOK: lambda r: num(r["tokens"]),
    }
    local_feats = {
        **base,
        "energy |E|/35 (as run)": en_as_run,
        "rcd_inv 1-min(2*RCD,1) (as run)": rcd_inv_as_run,
        "0.3*var + 0.2*decay (derived from hri)":
            lambda r: raw_from_hri(r["hri_f"]) - 0.2 * abs(float(r["mean_logprob"]))
            - 0.15 * en_as_run(r) - 0.15 * rcd_inv_as_run(r),
        TOK: lambda r: num(r["tokens"]),
    }
    gold_feats = {
        HRI: lambda r: r["hri_f"],
        MEAN: lambda r: float(r["abs_mean_logprob"]),
        "logprob_variance": lambda r: float(r["logprob_variance"]),
        "confidence_decay": lambda r: float(r["confidence_decay"]),
        TOK: lambda r: num(r["tokens"]),
    }

    # The derivation used on the API CSV, checked where the true features are logged.
    deriv_err = max(abs(raw_from_hri(r["hri_f"]) - 0.3 * float(r["abs_mean_logprob"])
                        - 0.4 * float(r["logprob_variance"]) - 0.3 * float(r["confidence_decay"]))
                    for r in gold)
    print(f"derivation check on gold: max |derived - true 0.4var+0.3decay| = {deriv_err:.5f}")

    res = {
        "derivation_max_abs_error_on_gold": deriv_err,
        "api": run("api", api, api_feats, [("SBERT, all 100 (as reported)", everything),
                                           ("SBERT, 82 real answers", real_api),
                                           ("audit, 82 real answers", real_api)]),
        "local": run("local", local, local_feats, [("SBERT, all 100 (as reported)", everything),
                                                   ("SBERT, 20 real answers", real_local),
                                                   ("audit, 20 real answers", real_local)]),
        "gold": run("gold", gold, gold_feats, [("SBERT@0.35, 80 answers", [True] * 80),
                                               ("audit, 80 answers", [True] * 80)]),
    }
    cos = {"SBERT cosine": lambda r: r["cos"]}
    res["spearman"] = {
        "api (82 real answers)": corr_matrix(api, {**api_feats, **cos}),
        "local (20 real answers)": corr_matrix(local, {**local_feats, **cos}),
        "gold (80 answers)": corr_matrix(gold, {**gold_feats, **cos}),
    }
    for name, cm in res["spearman"].items():
        print(f"\nSpearman, {name} (n={cm['n']})")
        keys = list(cm["matrix"])
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                print(f"  {a[:30]:<30} ~ {b[:30]:<30} {f(cm['matrix'][a][b], 2)}")
    save("e1_ablation", res)

    panels = [(s, ln, d["aucs"]) for s in ("api", "local", "gold") for ln, d in res[s].items()]
    heights = [sum(1 for v in p[2].values() if v) for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 0.32 * sum(heights) + 1.2 * len(panels)),
                             gridspec_kw={"height_ratios": heights})
    for ax, (s, ln, aucs) in zip(axes, panels):
        names = [k for k, v in aucs.items() if v]
        est = np.array([aucs[k]["est"] for k in names])
        lo = np.array([aucs[k]["lo"] for k in names])
        hi = np.array([aucs[k]["hi"] for k in names])
        yy = np.arange(len(names))
        for i, k in enumerate(names):
            ax.errorbar(est[i], yy[i], xerr=[[est[i] - lo[i]], [hi[i] - est[i]]], fmt="o",
                        capsize=3, color="#dc2626" if k == HRI else "#2563eb")
        ax.axvline(0.5, ls="--", color="grey")
        ax.set_yticks(yy, names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_title(f"{s} | {ln}", fontsize=8, loc="left")
    axes[-1].set_xlabel("AUROC (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(FIG / "e1_auroc_forest.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, (name, cm) in zip(axes, res["spearman"].items()):
        keys = list(cm["matrix"])
        m = np.array([[cm["matrix"][a][b]["est"] if cm["matrix"][a][b] else np.nan
                       for b in keys] for a in keys])
        ax.imshow(m, vmin=-1, vmax=1, cmap="coolwarm")
        short = [k.split(" (")[0][:22] for k in keys]
        ax.set_xticks(range(len(keys)), short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(keys)), short, fontsize=8)
        for i in range(len(keys)):
            for j in range(len(keys)):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Spearman rho, {name}, n={cm['n']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e1_spearman.png", dpi=130)


if __name__ == "__main__":
    main()
