"""HalluScope HRI evaluation: AUROC, ROC, F1, threshold tuning.

Reads eval/results_{api,local}.csv (each must have model_match filled
by eval/match.py), treats model_match=='mismatch' as the positive
class (a hallucination event), and reports:
    - AUROC
    - ROC curve, saved to eval/roc_{mode}.png
    - Best-F1 threshold and its precision/recall/F1
    - Confusion matrix at the CURRENT HRI HIGH threshold (0.46 default)
    - Confusion matrix at the BEST-F1 threshold

Run from repo root:
    python3 eval/analyze.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVAL_DIR = ROOT / "eval"

from backend.app import config_store  # noqa: E402
from eval.stats import (  # noqa: E402
    bootstrap_auroc_ci,
    expected_calibration_error,
    brier_score,
    reliability_curve,
)

# Single source of truth — same threshold used for live classification and
# intervention (config/thresholds.json). No hardcoded 0.46 / 0.30 drift.
CURRENT_HIGH_THRESHOLD = config_store.risk_high()


def load_results(mode: str, path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Returns (y_true, y_score, raw_rows) where positive=hallucination."""
    path = path or (EVAL_DIR / f"results_{mode}.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    y_true:  List[int] = []
    y_score: List[float] = []
    for r in rows:
        if r.get("model_match") not in ("match", "mismatch"):
            raise RuntimeError(
                f"row {r.get('id')} has model_match={r.get('model_match')!r}; "
                "did you run eval/match.py first?"
            )
        y_true.append(1 if r["model_match"] == "mismatch" else 0)
        y_score.append(float(r["hri"]))
    return np.array(y_true), np.array(y_score), rows


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray
                      ) -> Tuple[float, float, float, float]:
    """Sweep the unique scores and return (thr, p, r, f1) maximising F1."""
    candidates = sorted(set(y_score.tolist()))
    best = (0.5, 0.0, 0.0, -1.0)
    for thr in candidates:
        y_pred = (y_score >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best[3]:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            best = (float(thr), float(p), float(r), float(f1))
    return best


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"thr": thr, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(p), "recall": float(r), "f1": float(f1)}


def plot_roc(mode: str, y_true: np.ndarray, y_score: np.ndarray,
             auroc: float, best_thr: float, current_thr: float,
             out_path: Path) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find FPR/TPR at the two operating points
    def point_at(thr):
        # roc_curve thresholds are decreasing; pick the closest one
        i = int(np.argmin(np.abs(thresholds - thr)))
        return fpr[i], tpr[i]

    bx, by = point_at(best_thr)
    cx, cy = point_at(current_thr)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#3b82f6", linewidth=2,
            label=f"HalluScope HRI (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--",
            linewidth=1, label="chance")
    ax.scatter([bx], [by], color="#10b981", s=80, zorder=5,
               label=f"best-F1 thr = {best_thr:.3f}")
    ax.scatter([cx], [cy], color="#ef4444", s=60, marker="x", zorder=5,
               label=f"current HIGH thr = {current_thr:.2f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{mode} mode — AUROC = {auroc:.3f}")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _delong_vs_baselines(rows: List[dict], y_true: np.ndarray,
                         hri: np.ndarray) -> List[dict]:
    """For every ``baseline_<name>`` score column, DeLong-test HRI vs it.

    Returns a list of {baseline, auroc_hri, auroc_baseline, p_value}. Empty if
    no baseline columns are present.
    """
    from eval.stats import delong_test

    if not rows:
        return []
    bcols = [c for c in rows[0].keys() if c.startswith("baseline_")]
    out: List[dict] = []
    for col in bcols:
        try:
            bscore = np.array([float(r[col]) for r in rows])
        except (TypeError, ValueError):
            continue  # column not fully populated
        a_hri, a_base, p = delong_test(y_true, hri, bscore)
        out.append({"baseline": col, "auroc_hri": a_hri,
                    "auroc_baseline": a_base, "p_value": p})
        print(f"  DeLong HRI vs {col:>28}: "
              f"AUROC {a_hri:.3f} vs {a_base:.3f}  p={p:.4f}")
    return out


def plot_reliability(mode: str, y_true: np.ndarray, y_score: np.ndarray,
                     ece: float, out_path: Path, n_bins: int = 10) -> None:
    centers, acc, conf, counts = reliability_curve(y_true, y_score, n_bins)
    nz = counts > 0
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="perfect calibration")
    ax.plot(conf[nz], acc[nz], "o-", color="#3b82f6",
            label=f"HRI (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted HRI (confidence)")
    ax.set_ylabel("Empirical hallucination rate (accuracy)")
    ax.set_title(f"{mode} mode — reliability diagram")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def analyze_mode(mode: str, current_high: float = CURRENT_HIGH_THRESHOLD,
                 csv_path: Path | None = None) -> dict:
    y_true, y_score, rows = load_results(mode, csv_path)
    n_total = len(y_true)
    n_hallu = int(y_true.sum())
    n_corr  = n_total - n_hallu

    # Guard: AUROC needs both classes
    if n_corr == 0 or n_hallu == 0:
        print(f"\n=== {mode.upper()} mode ===")
        print(f"Samples: {n_total}  ({n_hallu} hallucination, {n_corr} correct)")
        print("ERROR: only one class present; AUROC undefined.")
        return {"mode": mode, "auroc": None, "error": "single class"}

    auroc = float(roc_auc_score(y_true, y_score))
    ci_lo, _, ci_hi = bootstrap_auroc_ci(y_true, y_score)

    # Calibration: HRI is already a [0,1] score, so ECE/Brier/reliability apply.
    ece = expected_calibration_error(y_true, y_score)
    brier = brier_score(y_true, y_score)

    best_thr, best_p, best_r, best_f1 = best_f1_threshold(y_true, y_score)
    cur = metrics_at(y_true, y_score, current_high)

    out_path = EVAL_DIR / f"roc_{mode}.png"
    plot_roc(mode, y_true, y_score, auroc, best_thr, current_high, out_path)

    rel_path = EVAL_DIR / f"reliability_{mode}.png"
    plot_reliability(mode, y_true, y_score, ece, rel_path)

    # DeLong HRI-vs-baseline comparisons (only if baseline_* columns present).
    delong_results = _delong_vs_baselines(rows, y_true, y_score)

    summary = f"""
=== {mode.upper()} mode ===
Samples:               {n_total}   ({n_hallu} hallucination, {n_corr} correct)
HRI distribution:      mean={statistics.mean(y_score):.3f}  std={statistics.stdev(y_score):.3f}  min={min(y_score):.3f}  max={max(y_score):.3f}
AUROC:                 {auroc:.3f}   (95% bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}])
Calibration:           ECE={ece:.3f}  Brier={brier:.3f}
Best-F1 threshold:     {best_thr:.3f}   (precision={best_p:.3f}, recall={best_r:.3f}, F1={best_f1:.3f})
Current HIGH threshold ({current_high:.2f}): precision={cur['precision']:.3f}, recall={cur['recall']:.3f}, F1={cur['f1']:.3f}
                              TP={cur['tp']}  FP={cur['fp']}  FN={cur['fn']}  TN={cur['tn']}
ROC plot saved:        {out_path.relative_to(ROOT)}
Reliability plot:      {rel_path.relative_to(ROOT)}"""
    print(summary)

    if auroc < 0.55:
        print("⚠️  AUROC near chance — HRI does not separate this dataset.")
        print("    Consider: (a) the prompt format is too easy/hard, (b) the")
        print("    HRI weights need refitting, (c) the sample is too small.")

    return {
        "mode": mode, "n_total": n_total,
        "n_hallu": n_hallu, "n_corr": n_corr,
        "auroc": auroc,
        "auroc_ci95": [ci_lo, ci_hi],
        "ece": ece,
        "brier": brier,
        "reliability_path": str(rel_path.relative_to(ROOT)),
        "delong_vs_baselines": delong_results,
        "hri_mean": statistics.mean(y_score),
        "hri_std": statistics.stdev(y_score),
        "best_thr": best_thr, "best_precision": best_p,
        "best_recall": best_r, "best_f1": best_f1,
        "current_high_threshold": current_high,
        "at_current": cur,
        "roc_path": str(out_path.relative_to(ROOT)),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None,
                    help="Analyse a single arbitrary results CSV (e.g. a test "
                         "split). Requires --mode. Default: the conventional "
                         "eval/results_{api,local}.csv pair.")
    ap.add_argument("--mode", choices=["api", "local"], default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Summary JSON path (default eval/summary.json; pick a "
                         "new path to avoid overwriting the committed summary).")
    args = ap.parse_args()

    results = {}
    if args.csv:
        if not args.mode:
            print("ERROR: --csv requires --mode", file=sys.stderr)
            return 2
        results[args.mode] = analyze_mode(args.mode, csv_path=args.csv)
    else:
        for mode in ("api", "local"):
            path = EVAL_DIR / f"results_{mode}.csv"
            if not path.exists():
                print(f"\n[skip] {path} not found")
                continue
            results[mode] = analyze_mode(mode)

    out_json = args.out or (EVAL_DIR / "summary.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSummary JSON saved: {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
