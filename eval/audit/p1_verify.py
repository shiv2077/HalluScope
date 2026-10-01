"""Phase 1: recompute the README / eval/REPORT.md numbers from the committed files.

Run: python3 eval/audit/p1_verify.py   -> eval/audit/out/p1_verify.json
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from common import ROOT, auc_ci, boot, col, f, load_gold, load_mode, read_csv, save, wilson


def at(y, s, t):
    flag = s >= t
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    p = tp / max(int(flag.sum()), 1)
    r = tp / max(int(y.sum()), 1)
    return {"thr": float(t), "flagged": int(flag.sum()), "flag_rate": wilson(int(flag.sum()), len(y)),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": wilson(tp, int(flag.sum())), "recall": wilson(tp, int(y.sum())),
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def best_f1(y, s):
    return max((at(y, s, t) for t in sorted(set(s.tolist()))), key=lambda d: d["f1"])


def main():
    out = {}
    for mode in ("api", "local"):
        rows = load_mode(mode)
        y = col(rows, "y").astype(int)
        s = col(rows, "hri_f")
        sup = np.array([r["suppressed"] for r in rows])
        m = {
            "n": len(y), "pos": int(y.sum()), "neg": int(len(y) - y.sum()),
            "auroc_as_reported": auc_ci(y, s),
            "best_f1": best_f1(y, s),
            "at_thresholds": [at(y, s, t) for t in (0.086, 0.27, 0.30, 0.40, 0.46)],
            "mean_hri_match": boot(np.mean, s[y == 0]),
            "mean_hri_mismatch": boot(np.mean, s[y == 1]),
            "max_hri_match": float(s[y == 0].max()),
            "label_x_answer": {f"{lab}|{'suppressed' if su else 'real'}": c for (lab, su), c
                               in Counter(zip((r["model_match"] for r in rows), sup)).items()},
            "min_hri_suppressed": float(s[sup].min()),
            "max_hri_real": float(s[~sup].max()),
            "risk_level_counts": Counter(r["risk_level"] for r in rows),
            "intervened_equals_suppressed": all(
                (r["intervened"] == "True") == r["suppressed"] for r in rows),
            "columns_logged": list(read_csv(ROOT / "eval" / f"results_{mode}.csv")[0])[:13],
        }
        wrong = [r for r in rows if r["judge"] == "I"]  # audit: real answers judged incorrect
        m["audit_incorrect_hri_below"] = {
            str(t): wilson(sum(r["hri_f"] < t for r in wrong), len(wrong)) for t in (0.15, 0.30)}
        if mode == "local":
            e = np.abs(col(rows, "mean_semantic_energy"))
            rcd = col(rows, "mean_rcd")
            m["raw_energy_all_negative"] = bool((col(rows, "mean_semantic_energy") < 0).all())
            m["energy_norm_as_run_range"] = [float(e.min() / 35), float(e.max() / 35)]
            m["energy_norm_clamped_at_1"] = int((e / 35 >= 1).sum())
            m["rcd_range"] = [float(rcd.min()), float(rcd.max())]
            m["rcd_inv_as_run_zero_rows"] = int((1 - np.minimum(rcd * 2, 1) == 0).sum())
        out[mode] = m
        print(f"\n== {mode}: n={m['n']} pos={m['pos']} neg={m['neg']} "
              f"AUROC {f(m['auroc_as_reported'])}")
        print("  label x answer:", m["label_x_answer"])
        print(f"  min HRI suppressed {m['min_hri_suppressed']}  max HRI real {m['max_hri_real']}")
        b = m["best_f1"]
        print(f"  best-F1 thr {b['thr']} P={b['precision']['est']:.3f} "
              f"R={b['recall']['est']:.3f} F1={b['f1']:.3f}")
        for d in m["at_thresholds"]:
            print(f"  thr {d['thr']}: flagged {d['flagged']} P {f(d['precision'])} "
                  f"R {f(d['recall'])} F1 {d['f1']:.3f} fp={d['fp']}")
        print(f"  mean HRI match {f(m['mean_hri_match'])} mismatch {f(m['mean_hri_mismatch'])}")
        if mode == "local":
            print("  energy_norm range", m["energy_norm_as_run_range"], "rcd_inv zero rows",
                  m["rcd_inv_as_run_zero_rows"], "of", m["n"])

    wrong = [r for r in load_gold() if r["judge"] == "I"]
    out["gold_audit_incorrect_hri_below"] = {
        str(t): wilson(sum(r["hri_f"] < t for r in wrong), len(wrong)) for t in (0.15, 0.30)}
    print("audit-incorrect answers scored below 0.15 / 0.30:",
          {k: {t: f(v, 2) for t, v in out[k]["audit_incorrect_hri_below"].items()} for k in ("api", "local")},
          "gold", {t: f(v, 2) for t, v in out["gold_audit_incorrect_hri_below"].items()})

    v1 = read_csv(ROOT / "eval" / "results_api_v1_candidate_eval.csv")
    yv = np.array([r["label"] != "correct" for r in v1], dtype=int)
    sv = np.array([float(r["hri"]) for r in v1])
    out["v1_candidate_eval"] = {
        "labels": Counter(r["label"] for r in v1), "at_0.46": at(yv, sv, 0.46),
        "mean_hri_correct": boot(np.mean, sv[yv == 0]),
        "mean_hri_hallucinated": boot(np.mean, sv[yv == 1]), "auroc": auc_ci(yv, sv)}
    v = out["v1_candidate_eval"]
    print(f"v1: {v['labels']} F1@0.46 {v['at_0.46']['f1']:.3f} mean HRI correct {f(v['mean_hri_correct'])} "
          f"halluc {f(v['mean_hri_hallucinated'])} AUROC {f(v['auroc'])}")

    d = ROOT / "eval" / "datasets"

    def qs(name):
        return {r["question"] for r in json.loads((d / name).read_text())}

    sub = qs("halluveval_qa_questions_100.json")
    out["splits"] = {
        "subset_in_train": len(sub & qs("halueval_qa_train.json")),
        "subset_in_test": len(sub & qs("halueval_qa_test.json")),
        "results_files": sorted(p.name for p in (ROOT / "eval").glob("results*.csv")),
    }
    gold_dir = ROOT / "eval" / "gold"
    out["gold"] = {
        "human_labels_filled_A": sum(bool(r["human_label"]) for r in read_csv(gold_dir / "gold_annotatorA.csv")),
        "human_labels_filled_B": sum(bool(r["human_label"]) for r in read_csv(gold_dir / "gold_annotatorB.csv")),
        "model_match_filled": sum(bool(r["model_match"]) for r in read_csv(gold_dir / "gold_results_api.csv")),
    }
    cfg = ROOT / "config"
    out["config"] = {
        "weights_source": json.loads((cfg / "weights.json").read_text())["source"],
        "grader_sbert_threshold": json.loads((cfg / "grader.json").read_text())["sbert_threshold"],
        "operating_point_used_fallback": json.loads(
            (ROOT / "eval" / "operating_point.json").read_text())["used_fallback"],
        "weights_fitted_file_exists": (ROOT / "eval" / "weights_fitted.json").exists(),
    }
    try:
        from eval.build_subset import _legacy_subset, _load_hf

        hf = _load_hf("halueval_qa")
        committed = json.loads((d / "halluveval_qa_questions_100.json").read_text())
        out["subset_regenerates_identically"] = _legacy_subset(hf, 100, 42, "halueval_qa") == committed
        out["halueval_rows"] = len(hf)
    except Exception as exc:  # offline cache missing
        out["subset_regenerates_identically"] = f"not checked: {exc!r}"
    print("\nsplits", out["splits"], "\ngold", out["gold"], "\nconfig", out["config"])
    print("subset regenerates identically:", out["subset_regenerates_identically"])
    save("p1_verify", out)


if __name__ == "__main__":
    main()
