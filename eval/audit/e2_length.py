"""E2: is HRI measuring answer length?

Token counts are re-tokenized from the logged answer text (o200k_base for
gpt-4o-mini, TinyLlama tokenizer for local). Suppressed rows have no answer text
and are excluded. Generation was capped at max_tokens=80.
Run: python3 eval/audit/e2_length.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import (FIG, auc_ci, f, load_gold, load_mode, raw_from_hri, save,  # noqa: E402
                    spearman_ci, wilson)


def absmean(r):
    return abs(float(r["mean_logprob"]))


LABELS = {"SBERT label (1=mismatch)": lambda r: r["y"], "audit label (1=incorrect)": lambda r: r["ya"]}


def main():
    api = [r for r in load_mode("api") if not r["suppressed"]]
    local = [r for r in load_mode("local") if not r["suppressed"]]
    gold = load_gold()
    sets = {
        "api (82 real answers)": (api, {
            "HRI": lambda r: r["hri_f"], "|mean logprob|": absmean,
            "0.4*var+0.3*decay (derived)": lambda r: raw_from_hri(r["hri_f"]) - 0.3 * absmean(r),
            "SBERT cosine": lambda r: r["cos"], **LABELS}),
        "local (20 real answers)": (local, {
            "HRI": lambda r: r["hri_f"], "|mean logprob|": absmean,
            "energy |E|/35 (as run)": lambda r: min(abs(float(r["mean_semantic_energy"])) / 35, 1),
            "SBERT cosine": lambda r: r["cos"], **LABELS}),
        "gold (80 answers, current code)": (gold, {
            "HRI": lambda r: r["hri_f"], "|mean logprob|": absmean,
            "logprob_variance": lambda r: float(r["logprob_variance"]),
            "confidence_decay": lambda r: float(r["confidence_decay"]),
            "SBERT cosine": lambda r: r["cos"], **LABELS}),
    }
    res = {}
    for name, (rows, feats) in sets.items():
        t = np.array([r["tokens"] for r in rows], float)
        hri = np.array([r["hri_f"] for r in rows])
        am = np.array([absmean(r) for r in rows])
        d = {
            "n": len(rows),
            "tokens": {"min": int(t.min()), "median": float(np.median(t)), "max": int(t.max()),
                       "n_at_or_near_cap_ge78": int((t >= 78).sum())},
            "spearman_tokens_vs": {k: spearman_ci(t, [fn(r) for r in rows]) for k, fn in feats.items()},
            "length_alone_auroc": {k: auc_ci(np.array([fn(r) for r in rows]), t) for k, fn in LABELS.items()},
            "buckets": [],
        }
        edges = np.quantile(t, [1 / 3, 2 / 3])
        b = np.searchsorted(edges, t, side="right")
        for i in range(3):
            m = b == i
            if not m.any():
                continue
            e = {"bucket": i, "tokens": [int(t[m].min()), int(t[m].max())], "n": int(m.sum())}
            for k, fn in LABELS.items():
                yv = np.array([fn(r) for r in rows])[m]
                e[k] = {"pos_rate": wilson(int(yv.sum()), int(m.sum())),
                        "auroc_hri": auc_ci(yv, hri[m]), "auroc_abs_mean_logprob": auc_ci(yv, am[m])}
            d["buckets"].append(e)
        res[name] = d

        print(f"\n== {name}: tokens {d['tokens']}")
        for k, v in d["spearman_tokens_vs"].items():
            print(f"  rho(tokens, {k:<30}) {f(v, 2)}")
        for k, v in d["length_alone_auroc"].items():
            print(f"  AUROC of tokens alone vs {k:<26} {f(v)}")
        for e in d["buckets"]:
            for k in LABELS:
                x = e[k]
                print(f"  bucket {e['bucket']} tokens {e['tokens']} n={e['n']} {k[:5]} "
                      f"pos_rate {f(x['pos_rate'], 2)} AUROC HRI {f(x['auroc_hri'])} "
                      f"|mean| {f(x['auroc_abs_mean_logprob'])}")
    save("e2_length", res)

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for row, (name, (rows, _)) in zip(axes, sets.items()):
        t = [r["tokens"] for r in rows]
        c = ["#dc2626" if r["ya"] else "#16a34a" for r in rows]
        for ax, key, lab in ((row[0], "hri_f", "HRI"), (row[1], "cos", "SBERT cosine")):
            ax.scatter(t, [r[key] for r in rows], c=c, s=18, alpha=0.8)
            ax.set_xlabel("answer tokens (re-tokenized)")
            ax.set_ylabel(lab)
            rho = res[name]["spearman_tokens_vs"]["HRI" if key == "hri_f" else "SBERT cosine"]
            ax.set_title(f"{name}: rho = {f(rho, 2)}", fontsize=9)
        row[1].axhline(0.35, ls="--", color="grey")
    axes[0][0].scatter([], [], c="#dc2626", label="audit: incorrect")
    axes[0][0].scatter([], [], c="#16a34a", label="audit: correct")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e2_length_scatter.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ["api (82 real answers)", "gold (80 answers, current code)"]):
        for j, (k, colr) in enumerate(zip(LABELS, ("#2563eb", "#dc2626"))):
            xs, est, lo, hi, ticks = [], [], [], [], []
            for e in res[name]["buckets"]:
                a = e[k]["auroc_hri"]
                ticks.append(f"{e['tokens'][0]}-{e['tokens'][1]} tok\nn={e['n']}")
                if a:
                    xs.append(e["bucket"] + (j - 0.5) * 0.2)
                    est.append(a["est"])
                    lo.append(a["est"] - a["lo"])
                    hi.append(a["hi"] - a["est"])
            ax.errorbar(xs, est, yerr=[lo, hi], fmt="o", color=colr, capsize=3, label=k)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
        ax.axhline(0.5, ls="--", color="grey")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("AUROC of HRI within bucket")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "e2_bucket_auroc.png", dpi=130)


if __name__ == "__main__":
    main()
