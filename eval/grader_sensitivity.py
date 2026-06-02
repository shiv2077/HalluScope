"""SBERT grader-threshold sensitivity sweep.

The match/mismatch ground truth for the benchmark is derived from a single
SBERT cosine threshold (config/grader.json -> sbert_threshold). Because that
threshold directly determines the positive (hallucination) class, every
reported AUROC/F1 depends on it. This tool shows the HUMAN how the split moves
across a range of thresholds so the choice can be justified, instead of being
silently tuned on the eval labels.

It DOES NOT pick a value and DOES NOT modify the input CSV. It prints a table
and (optionally) saves it.

Usage:
    # Sweep over the default thresholds {0.45..0.70} on a results CSV that has
    # model_answer + reference_answer columns:
    python3 eval/grader_sensitivity.py --csv eval/results_api.csv

    # Save the sweep table:
    python3 eval/grader_sensitivity.py --csv eval/results_api.csv \
        --out eval/grader_sensitivity_api.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The sweep grid requested for sensitivity analysis. Pure reporting — the
# human chooses; this script never writes a threshold back to config.
DEFAULT_SWEEP: List[float] = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def sweep_counts(sims: Sequence[float], thresholds: Sequence[float]) -> List[Dict]:
    """For each threshold, count match (cos >= thr) vs mismatch (cos < thr).

    Pure function — no model needed. ``mismatch`` is the positive
    (hallucination) class, so ``mismatch_rate`` is the positive-class rate.
    """
    n_total = len(sims)
    rows: List[Dict] = []
    for thr in thresholds:
        n_match = sum(1 for s in sims if s >= thr)
        n_mismatch = n_total - n_match
        rows.append({
            "threshold": float(thr),
            "n_total": n_total,
            "n_match": n_match,
            "n_mismatch": n_mismatch,
            "match_rate": (n_match / n_total) if n_total else 0.0,
            "mismatch_rate": (n_mismatch / n_total) if n_total else 0.0,
        })
    return rows


def _cosines_from_csv(csv_path: Path) -> List[float]:
    """Recompute (or read) SBERT cosine for each row of a results CSV."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"{csv_path} is empty")
    # Prefer a precomputed sbert_cos column; otherwise compute it.
    if "sbert_cos" in rows[0] and all(r.get("sbert_cos") for r in rows):
        return [float(r["sbert_cos"]) for r in rows]
    if "model_answer" not in rows[0] or "reference_answer" not in rows[0]:
        raise RuntimeError(
            f"{csv_path} lacks sbert_cos and model_answer/reference_answer; "
            "cannot compute cosine."
        )
    from eval.match import batch_cosine
    return batch_cosine(
        [r["model_answer"] for r in rows],
        [r["reference_answer"] for r in rows],
    )


def _print_table(rows: List[Dict]) -> None:
    print(f"  {'threshold':>9}  {'n_match':>8}  {'n_mismatch':>10}  "
          f"{'match%':>7}  {'mismatch%(pos)':>14}")
    for r in rows:
        print(f"  {r['threshold']:>9.2f}  {r['n_match']:>8d}  "
              f"{r['n_mismatch']:>10d}  {r['match_rate']*100:>6.1f}%  "
              f"{r['mismatch_rate']*100:>13.1f}%")
    print()
    print("This is a sensitivity table, NOT a recommendation. Choose the "
          "grader threshold against a manually-labelled gold subset, then "
          "set config/grader.json -> sbert_threshold accordingly.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="Results CSV with sbert_cos or model_answer+reference_answer.")
    ap.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_SWEEP)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to save the sweep table as CSV.")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 2

    sims = _cosines_from_csv(args.csv)
    rows = sweep_counts(sims, args.thresholds)

    print(f"\n=== SBERT grader sensitivity for {args.csv} (n={len(sims)}) ===")
    _print_table(rows)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved sweep table: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
