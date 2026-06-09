"""Calibrate the SBERT grader threshold against HUMAN answer-equivalence labels.

The grader decides whether a model's free-form answer is equivalent to the
dataset reference. Its cosine threshold must be set to agree with HUMANS, not
to fit a model's output histogram (which makes it model-specific and circular).

Given a gold CSV that a human has labelled — columns ``model_answer``,
``reference_answer``, ``human_label`` ∈ {correct, incorrect} — this tool:

  1. computes SBERT cosine(model_answer, reference_answer) per row,
  2. sweeps thresholds 0.40..0.80 (step 0.01) and, per threshold, reports
     agreement (accuracy) with the human labels, F1 (positive class =
     'incorrect', the hallucination class), and Cohen's kappa,
  3. prints the threshold that MAXIMISES human agreement (NOT the one that
     looks bimodal),
  4. if two annotator files are given, prints inter-annotator kappa,
  5. saves a small grader_calibration_report.csv (a calibration artifact —
     NOT a results/summary/REPORT file).

It RUNS on the human's real labels and fabricates nothing. It does NOT modify
config/grader.json; it prints the exact command to do so after review.

Usage (human runs this on their hand-labelled file(s)):
    python3 eval/grader_calibration.py --gold eval/gold/gold_annotatorA.csv
    python3 eval/grader_calibration.py \
        --gold eval/gold/gold_annotatorA.csv \
        --gold-b eval/gold/gold_annotatorB.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Threshold grid 0.40..0.80 step 0.01 (inclusive).
THRESHOLDS: List[float] = [round(0.40 + 0.01 * i, 2) for i in range(41)]

VALID_LABELS = {"correct", "incorrect"}


# ── Pure scoring ─────────────────────────────────────────────────────
def _predict(cosines: Sequence[float], threshold: float) -> List[str]:
    """grader predicts 'correct' when cosine >= threshold, else 'incorrect'."""
    return ["correct" if c >= threshold else "incorrect" for c in cosines]


def sweep_agreement(
    cosines: Sequence[float],
    human_labels: Sequence[str],
    thresholds: Sequence[float] = THRESHOLDS,
) -> List[Dict]:
    """Per-threshold agreement(accuracy), F1 (positive='incorrect'), kappa.

    Pure: takes precomputed cosines so it is testable without SBERT.
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    rows: List[Dict] = []
    for thr in thresholds:
        pred = _predict(cosines, thr)
        acc = float(accuracy_score(human_labels, pred))
        f1 = float(f1_score(human_labels, pred, pos_label="incorrect", zero_division=0))
        kappa = float(
            cohen_kappa_score(human_labels, pred, labels=["correct", "incorrect"])
        )
        rows.append(
            {"threshold": float(thr), "accuracy": acc, "f1": f1, "kappa": kappa}
        )
    return rows


def best_threshold(sweep_rows: List[Dict]) -> Dict:
    """Row maximising human agreement (accuracy); ties broken by higher kappa,
    then by the smaller threshold."""
    return sorted(
        sweep_rows,
        key=lambda r: (-r["accuracy"], -r["kappa"], r["threshold"]),
    )[0]


def inter_annotator_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(
        cohen_kappa_score(
            list(labels_a), list(labels_b), labels=["correct", "incorrect"]
        )
    )


# ── IO ───────────────────────────────────────────────────────────────
def _load_gold(path: Path) -> List[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    for col in ("model_answer", "reference_answer", "human_label"):
        if col not in rows[0]:
            raise SystemExit(f"{path} is missing required column '{col}'.")
    bad = [
        r
        for r in rows
        if (r.get("human_label") or "").strip().lower() not in VALID_LABELS
    ]
    if bad:
        raise SystemExit(
            f"{path}: {len(bad)} rows have an unfilled/invalid human_label "
            f"(must be one of {sorted(VALID_LABELS)}). Label every row first."
        )
    if any(not (r.get("model_answer") or "").strip() for r in rows):
        raise SystemExit(
            f"{path}: some rows have an empty model_answer. Fill them with the "
            f"model's REAL answers (run_benchmark on the gold questions, then "
            f"make_gold_template.py --fill-from) before calibrating."
        )
    return rows


def _cosines(rows: List[dict]) -> List[float]:
    from eval.match import batch_cosine

    return batch_cosine(
        [r["model_answer"] for r in rows], [r["reference_answer"] for r in rows]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold", type=Path, required=True, help="Hand-labelled gold CSV (annotator A)."
    )
    ap.add_argument(
        "--gold-b",
        type=Path,
        default=None,
        help="Optional second annotator's CSV for inter-annotator kappa.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval" / "gold" / "grader_calibration_report.csv",
    )
    args = ap.parse_args()

    rows = _load_gold(args.gold)
    labels = [(r["human_label"] or "").strip().lower() for r in rows]
    cosines = _cosines(rows)

    sweep = sweep_agreement(cosines, labels, THRESHOLDS)
    best = best_threshold(sweep)

    print(
        f"\n=== Grader calibration vs human labels ({args.gold.name}, "
        f"n={len(rows)}) ==="
    )
    print(
        f"  human label balance: "
        f"correct={labels.count('correct')}  incorrect={labels.count('incorrect')}"
    )
    print(f"  {'thr':>5}  {'agreement':>9}  {'F1(incorr)':>10}  {'kappa':>6}")
    for r in sweep:
        mark = (
            "  <- max agreement"
            if abs(r["threshold"] - best["threshold"]) < 1e-9
            else ""
        )
        # print a readable subset: every 0.05 plus the best
        if abs((r["threshold"] * 100) % 5) < 1e-6 or mark:
            print(
                f"  {r['threshold']:>5.2f}  {r['accuracy']:>9.3f}  "
                f"{r['f1']:>10.3f}  {r['kappa']:>6.3f}{mark}"
            )

    print(
        f"\nRECOMMENDED threshold (max human agreement): {best['threshold']:.2f}  "
        f"(agreement={best['accuracy']:.3f}, F1={best['f1']:.3f}, "
        f"kappa={best['kappa']:.3f})"
    )

    if args.gold_b is not None:
        rows_b = _load_gold(args.gold_b)
        labels_b = [(r["human_label"] or "").strip().lower() for r in rows_b]
        if len(labels_b) != len(labels):
            print("\n[warn] annotator files differ in length; skipping IAA.")
        else:
            iaa = inter_annotator_kappa(labels, labels_b)
            print(f"\nInter-annotator agreement (Cohen's kappa A vs B): {iaa:.3f}")
            print(
                "  (Interpret: <0.40 poor, 0.40-0.60 moderate, 0.60-0.80 "
                "substantial, >0.80 almost perfect. Low kappa => the labelling "
                "guideline is ambiguous; fix it before trusting the threshold.)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "accuracy", "f1", "kappa"])
        w.writeheader()
        w.writerows(sweep)
    print(f"\nSaved sweep table: {args.out}")

    print(
        "\nThis tool does NOT edit config/grader.json. After reviewing the "
        "kappa above, set the threshold yourself, e.g.:"
    )
    print(
        f"  python3 -c \"import json,pathlib; p=pathlib.Path('config/grader.json'); "
        f"d=json.loads(p.read_text()); d['sbert_threshold']={best['threshold']:.2f}; "
        f'p.write_text(json.dumps(d, indent=2))"'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
