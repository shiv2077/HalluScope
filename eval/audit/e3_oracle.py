"""E3: is the SBERT@0.35 correctness label trustworthy?

Writes eval/audit/oracle_sample.csv (100 API rows in a seed-42 random order; the API
eval set has exactly 100 rows) plus local and gold companions, compares the SBERT
label with the audit judgements (eval/audit/judgements.py), and sweeps the SBERT
threshold from 0.25 to 0.75.
Run: python3 eval/audit/e3_oracle.py
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

from common import (AUDIT, FIG, SEED, auc_ci, f, kappa_ci, load_gold, load_mode, save,  # noqa: E402
                    wilson)
from judgements import AMBIGUOUS, NOTES  # noqa: E402

GRID = np.round(np.arange(0.25, 0.7501, 0.05), 2)
FINE = np.round(np.arange(0.25, 0.7501, 0.01), 2)
WORD = {"C": "correct", "I": "incorrect", "U": "unjudgeable: answer replaced by intervention text"}


def write_sample(path, rows, set_name):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question", "model_answer", "reference_answer", "sbert_cos",
                    "sbert_label_at_0.35", "audit_judgement", "audit_ambiguous", "audit_note", "hri"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["model_answer"], r["reference_answer"],
                        f"{r['cos']:.4f}", "mismatch" if r["y"] else "match", WORD[r["judge"]],
                        r["id"] in AMBIGUOUS, NOTES[set_name].get(r["id"], ""), r["hri"]])


def agreement(rows):
    rows = [r for r in rows if r["judge"] != "U"]
    s = np.array([r["y"] for r in rows])
    a = np.array([r["ya"] for r in rows])
    amb = np.array([r["id"] in AMBIGUOUS for r in rows])
    cm = confusion_matrix(s, a, labels=[0, 1])
    return {
        "n": len(rows),
        "confusion": {"sbert_match & audit_correct": int(cm[0, 0]),
                      "sbert_match & audit_incorrect": int(cm[0, 1]),
                      "sbert_mismatch & audit_correct": int(cm[1, 0]),
                      "sbert_mismatch & audit_incorrect": int(cm[1, 1])},
        "agreement": wilson(int((s == a).sum()), len(s)),
        "kappa": kappa_ci(s, a),
        "agreement_excluding_ambiguous": wilson(int((s == a)[~amb].sum()), int((~amb).sum())),
        "audit_correct": int((a == 0).sum()), "sbert_match": int((s == 0).sum()),
    }


def sweep(rows, grid, with_ci):
    cos = np.array([r["cos"] for r in rows])
    h = np.array([r["hri_f"] for r in rows])
    judged = np.array([r["judge"] != "U" for r in rows])
    a = np.array([r["ya"] for r in rows])
    out = []
    for t in grid:
        y = (cos < t).astype(int)
        both = 0 < y.sum() < len(y)
        d = {"thr": float(t), "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
             "auroc": (auc_ci(y, h) if with_ci else ({"est": float(roc_auc_score(y, h))} if both else None)),
             "agreement_with_audit": wilson(int((y == a)[judged].sum()), int(judged.sum())) if judged.any() else None}
        out.append(d)
    return out


def main():
    api = load_mode("api")
    local = load_mode("local")
    gold = load_gold()
    order = np.random.RandomState(SEED).permutation(len(api))
    write_sample(AUDIT / "oracle_sample.csv", [api[i] for i in order[:100]], "api")
    write_sample(AUDIT / "oracle_sample_local.csv", local, "local")
    write_sample(AUDIT / "oracle_sample_gold.csv", gold, "gold")

    res = {"agreement": {"api (82 judgeable of 100)": agreement(api),
                         "local (20 judgeable of 100)": agreement(local),
                         "gold (80)": agreement(gold)}}
    for k, v in res["agreement"].items():
        print(f"\n== {k}: agreement {f(v['agreement'])} kappa {f(v['kappa'])} "
              f"excl. ambiguous {f(v['agreement_excluding_ambiguous'])}")
        print("  ", v["confusion"], f"audit correct={v['audit_correct']} sbert match={v['sbert_match']}")

    try:  # the stored cosines are what the headline labels were built from; re-derive them
        from eval.match import batch_cosine

        res["sbert_recompute_max_abs_diff"] = {
            m: float(np.max(np.abs(np.round(batch_cosine([r["model_answer"] for r in rows],
                                                         [r["reference_answer"] for r in rows]), 4)
                                   - np.array([r["cos"] for r in rows]))))
            for m, rows in (("api", api), ("local", local))}
    except Exception as exc:
        res["sbert_recompute_max_abs_diff"] = f"not checked: {exc!r}"
    print("\nSBERT recompute max abs diff:", res["sbert_recompute_max_abs_diff"])

    sets = {"api, all 100 (as reported)": api,
            "api, 82 real answers": [r for r in api if not r["suppressed"]],
            "gold, 80 answers": gold,
            "local, all 100 (as reported)": local,
            "local, 20 real answers": [r for r in local if not r["suppressed"]]}
    res["sweep"] = {k: sweep(v, GRID, True) for k, v in sets.items()}
    res["sweep_fine_points"] = {k: sweep(v, FINE, False) for k, v in sets.items()}
    for k, rows in res["sweep"].items():
        print(f"\nSweep {k}")
        for d in rows:
            print(f"  thr {d['thr']:.2f} pos={d['pos']:>3} neg={d['neg']:>3} AUROC {f(d['auroc'])} "
                  f"agree {f(d['agreement_with_audit'], 2)}")
    save("e3_oracle", res)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (k, colr) in zip(["api, all 100 (as reported)", "api, 82 real answers", "gold, 80 answers"],
                         ("#dc2626", "#2563eb", "#16a34a")):
        fine = [(d["thr"], d["auroc"]["est"]) for d in res["sweep_fine_points"][k] if d["auroc"]]
        axes[0].plot(*zip(*fine), color=colr, lw=1, alpha=0.6)
        pts = [d for d in res["sweep"][k] if d["auroc"]]
        e = np.array([d["auroc"]["est"] for d in pts])
        axes[0].errorbar([d["thr"] for d in pts], e,
                         yerr=[e - [d["auroc"]["lo"] for d in pts], [d["auroc"]["hi"] for d in pts] - e],
                         fmt="o", color=colr, capsize=3, label=k)
        if k != "api, all 100 (as reported)":
            ag = [d for d in res["sweep"][k] if d["agreement_with_audit"]]
            g = np.array([d["agreement_with_audit"]["est"] for d in ag])
            axes[1].errorbar([d["thr"] for d in ag], g,
                             yerr=[g - [d["agreement_with_audit"]["lo"] for d in ag],
                                   [d["agreement_with_audit"]["hi"] for d in ag] - g],
                             fmt="o-", color=colr, capsize=3, label=k)
    for ax in axes:
        ax.axvline(0.35, ls="--", color="grey")
        ax.set_xlabel("SBERT cosine threshold defining 'match'")
        ax.legend(fontsize=8)
    axes[0].axhline(0.5, ls=":", color="grey")
    axes[0].set_ylabel("AUROC of HRI (95% CI)")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("agreement of SBERT label with audit label (95% Wilson CI)")
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "e3_threshold_sweep.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, rows) in zip(axes, (("api, 82 real answers", sets["api, 82 real answers"]),
                                       ("gold, 80 answers", gold))):
        bins = np.linspace(-0.1, 0.8, 28)
        ax.hist([r["cos"] for r in rows if r["judge"] == "C"], bins=bins, alpha=0.6,
                color="#16a34a", label="audit: correct")
        ax.hist([r["cos"] for r in rows if r["judge"] == "I"], bins=bins, alpha=0.6,
                color="#dc2626", label="audit: incorrect")
        ax.axvline(0.35, ls="--", color="grey")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("SBERT cosine(model answer, reference)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e3_cosine_by_audit_label.png", dpi=130)


if __name__ == "__main__":
    main()
