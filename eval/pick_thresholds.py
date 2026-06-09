"""Pick precision-floor operating point thresholds from API benchmark.

Best-F1 maximisation gave HIGH=0.086, which is too aggressive for live
deployment (would flag ~80% of normal traffic as HIGH risk because the
HalluEval calibration set is 62% adversarial-by-design). This script
sweeps the same data for the SMALLEST threshold that simultaneously
satisfies:

    precision      >= 0.85   (trustworthy when it does fire)
    recall         >= 0.50   (still catches half the hallucinations)
    flag_rate      <= 0.50   (doesn't fire on >50% of normal traffic)

If no threshold meets all three, falls back to (HIGH=0.30, LOW=0.15).

Run from repo root:
    python3 eval/pick_thresholds.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "eval" / "results_api.csv"

PRECISION_FLOOR = 0.85
RECALL_FLOOR = 0.50
FLAG_RATE_CEIL = 0.50

# Fallback if no threshold satisfies all three constraints
FALLBACK_HIGH = 0.30
FALLBACK_LOW = 0.15


def main() -> dict:
    rows = list(csv.DictReader(open(RESULTS_PATH, encoding="utf-8")))
    y_true = np.array([1 if r["model_match"] == "mismatch" else 0 for r in rows])
    y_score = np.array([float(r["hri"]) for r in rows])
    n_total = len(y_true)

    print(f"Loaded {n_total} rows from {RESULTS_PATH.relative_to(ROOT)}")
    print(f"  positives (hallucination): {int(y_true.sum())}")
    print(f"  negatives (correct)      : {int(n_total - y_true.sum())}")
    print()

    # Sweep 0.05..0.95 inclusive in 0.01 steps
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows_sweep = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        n_flag = int(y_pred.sum())
        if n_flag == 0:
            p = r = f1 = 0.0
        else:
            p = float(precision_score(y_true, y_pred, zero_division=0))
            r = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rows_sweep.append(
            {
                "thr": float(thr),
                "precision": p,
                "recall": r,
                "f1": f1,
                "flag_rate": n_flag / n_total,
                "n_flag": n_flag,
            }
        )

    # Find smallest threshold satisfying all three constraints
    qualifying = [
        s
        for s in rows_sweep
        if s["precision"] >= PRECISION_FLOOR
        and s["recall"] >= RECALL_FLOOR
        and s["flag_rate"] <= FLAG_RATE_CEIL
    ]

    if qualifying:
        chosen = min(qualifying, key=lambda s: s["thr"])
        used_fallback = False
    else:
        chosen = {
            "thr": FALLBACK_HIGH,
            "precision": None,
            "recall": None,
            "f1": None,
            "flag_rate": None,
            "n_flag": None,
        }
        used_fallback = True

    high = round(chosen["thr"], 2)
    low = round(max(0.10, high - 0.15), 2)

    # ── Report ─────────────────────────────────────────────────────
    print(
        f"Constraints: precision >= {PRECISION_FLOOR}, "
        f"recall >= {RECALL_FLOOR}, flag_rate <= {FLAG_RATE_CEIL}"
    )
    print(f"Qualifying thresholds: {len(qualifying)}")
    print()
    print("=== Sample of the sweep around the chosen point ===")
    print(
        f"  {'thr':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'flag%':>6}  {'qualifies':>10}"
    )
    # Print a window: from chosen-0.10 to chosen+0.10
    for s in rows_sweep:
        if abs(s["thr"] - high) <= 0.10:
            q = (
                "YES"
                if (
                    s["precision"] >= PRECISION_FLOOR
                    and s["recall"] >= RECALL_FLOOR
                    and s["flag_rate"] <= FLAG_RATE_CEIL
                )
                else "no"
            )
            marker = " <-- chosen" if abs(s["thr"] - high) < 1e-6 else ""
            print(
                f"  {s['thr']:>5.2f}  {s['precision']:>7.3f}  {s['recall']:>7.3f}  "
                f"{s['f1']:>7.3f}  {s['flag_rate']*100:>5.1f}%  {q:>10}{marker}"
            )

    print()
    if used_fallback:
        print("FALLBACK: no threshold met all three constraints.")
        print(f"  Using HIGH={FALLBACK_HIGH}, LOW={FALLBACK_LOW}")
    else:
        print("Selected operating point:")
        print(
            f"  HIGH = {high}   (precision={chosen['precision']:.3f}, "
            f"recall={chosen['recall']:.3f}, "
            f"flag_rate={chosen['flag_rate']*100:.1f}%)"
        )
        print(f"  LOW  = max(0.10, HIGH - 0.15) = {low}")

    out = {
        "constraints": {
            "precision_floor": PRECISION_FLOOR,
            "recall_floor": RECALL_FLOOR,
            "flag_rate_ceil": FLAG_RATE_CEIL,
        },
        "n_qualifying": len(qualifying),
        "used_fallback": used_fallback,
        "high_threshold": high,
        "low_threshold": low,
        "operating_point_metrics": (
            {
                k: chosen[k]
                for k in ("thr", "precision", "recall", "f1", "flag_rate", "n_flag")
            }
            if not used_fallback
            else None
        ),
    }
    out_path = ROOT / "eval" / "operating_point.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    main()
