"""Fit HRI combination weights by logistic regression on a TRAIN split.

This is what turns HalluScope's "learned combination" from a claim into a
fact: instead of hand-set heuristic weights, it fits

    P(hallucination) = sigmoid( sum_i w_i * feature_i + b )

on the training CSV's per-signal feature columns (with model_match filled by
eval/match.py), then writes the coefficients into config/weights.json so the
live engine (hri_engine.hri_from_features / compute_hri) uses them. A fitted
mode block carries its own sigmoid {scale: 1.0, bias: intercept}.

Hard rules honoured:
  * Fit on TRAIN only — never the test set you report on.
  * Writes weights (capability), prints the coefficients; it does NOT compute
    or record any AUROC/F1. The human evaluates on the held-out test split.

Usage (human runs these; the assistant never fabricates the outputs):
    # Fit the local 5-feature HRI on the local TRAIN results, apply to config:
    python3 eval/fit_weights.py --csv eval/results_local_train.csv \
        --mode local --apply

    # Dry run (print coefficients, write eval/weights_fitted.json, do NOT touch config):
    python3 eval/fit_weights.py --csv eval/results_api_train.csv --mode api

    # Restore the documented heuristic weights:
    python3 eval/fit_weights.py --heuristic --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app import config_store  # noqa: E402
from backend.app.risk.hri_engine import (  # noqa: E402
    FEATURE_NAMES_API,
    FEATURE_NAMES_LOCAL,
)
from backend.app.seeding import set_global_seed  # noqa: E402

CONFIG_WEIGHTS = ROOT / "config" / "weights.json"

# Documented heuristic defaults (mirror of the shipped config/weights.json) so
# --heuristic can restore them deterministically.
HEURISTIC_WEIGHTS = {
    "_doc": "Restored heuristic defaults (eval/fit_weights.py --heuristic).",
    "source": "heuristic-default",
    "sigmoid": {"scale": 10.0, "bias": -3.0},
    "api": {"logprob_variance": 0.4, "confidence_decay": 0.3, "abs_mean_logprob": 0.3},
    "local": {"logprob_variance": 0.30, "confidence_decay": 0.20,
              "abs_mean_logprob": 0.20, "semantic_energy": 0.15, "rcd_inv": 0.15},
    "prompt_risk": {
        "_doc": "Stage 1 aggregation weights.",
        "sigmoid": {"scale": 4.0, "bias": -2.0},
        "entity_rarity": 1.0, "time_risk": 1.0,
        "prompt_perplexity": 1.0, "quoted_title": 0.5,
    },
}

FEATURE_SETS = {"api": FEATURE_NAMES_API, "local": FEATURE_NAMES_LOCAL}


def build_xy(rows: List[dict], feature_names: Sequence[str]
             ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Assemble (X, y, n_used) from CSV rows.

    Rows missing any selected feature (None/empty) or a valid model_match are
    skipped. y = 1 for model_match == 'mismatch' (the positive/hallucination
    class), 0 for 'match'.
    """
    X: List[List[float]] = []
    y: List[int] = []
    for r in rows:
        if r.get("model_match") not in ("match", "mismatch"):
            continue
        vals = []
        ok = True
        for fn in feature_names:
            v = r.get(fn, "")
            if v is None or v == "":
                ok = False
                break
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        X.append(vals)
        y.append(1 if r["model_match"] == "mismatch" else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int), len(y)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int = 42
                 ) -> Tuple[np.ndarray, float]:
    """Fit logistic regression; return (coefs, intercept)."""
    from sklearn.linear_model import LogisticRegression

    set_global_seed(seed)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(X, y)
    return clf.coef_[0], float(clf.intercept_[0])


def to_weight_block(feature_names: Sequence[str], coefs: np.ndarray,
                    intercept: float) -> Dict:
    """Build a config weight block: per-feature weights + scale=1 / bias=intercept."""
    block: Dict = {fn: float(c) for fn, c in zip(feature_names, coefs)}
    block["sigmoid"] = {"scale": 1.0, "bias": float(intercept)}
    return block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, help="TRAIN results CSV (feature columns + model_match).")
    ap.add_argument("--mode", choices=list(FEATURE_SETS), default="local")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true",
                    help="Write config/weights.json (live). Default: dry-run to "
                         "eval/weights_fitted.json only.")
    ap.add_argument("--heuristic", action="store_true",
                    help="Restore the documented heuristic weights (ignores --csv).")
    args = ap.parse_args()

    if args.heuristic:
        target = CONFIG_WEIGHTS if args.apply else (ROOT / "eval" / "weights_fitted.json")
        target.write_text(json.dumps(HEURISTIC_WEIGHTS, indent=2))
        config_store.reload()
        print(f"Wrote heuristic weights -> {target}")
        return 0

    if not args.csv or not args.csv.exists():
        print("ERROR: --csv <train results> is required (or use --heuristic).",
              file=sys.stderr)
        return 2

    feats = FEATURE_SETS[args.mode]
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    X, y, n_used = build_xy(rows, feats)
    if n_used < 10 or len(set(y.tolist())) < 2:
        print(f"ERROR: need >=10 rows and both classes; got n={n_used}, "
              f"classes={sorted(set(y.tolist()))}.", file=sys.stderr)
        return 2

    coefs, intercept = fit_logistic(X, y, seed=args.seed)
    block = to_weight_block(feats, coefs, intercept)

    # Merge into the existing weights file, updating only this mode's block.
    current = json.loads(CONFIG_WEIGHTS.read_text())
    current[args.mode] = block
    current["source"] = f"logreg-fitted({args.mode}, n={n_used}, seed={args.seed})"

    print(f"Fitted {args.mode} HRI on {n_used} TRAIN rows (positives={int(y.sum())}):")
    for fn, c in zip(feats, coefs):
        print(f"    {fn:>20}: {c:+.4f}")
    print(f"    {'(intercept/bias)':>20}: {intercept:+.4f}")
    print("NOTE: no AUROC/F1 computed here. Evaluate on the held-out TEST split "
          "with eval/analyze.py.")

    target = CONFIG_WEIGHTS if args.apply else (ROOT / "eval" / "weights_fitted.json")
    target.write_text(json.dumps(current, indent=2))
    config_store.reload()
    if args.apply:
        print(f"APPLIED -> {target} (live engine now uses fitted {args.mode} weights).")
    else:
        print(f"Dry run -> {target}. Re-run with --apply to update config/weights.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
