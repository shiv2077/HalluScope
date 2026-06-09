"""HRI ablation: leave-one-out and single-signal-alone.

Operates on a TEST feature CSV (model_match filled) and a weight block (from
config/weights.json or a fitted weights file). Reports, for the chosen mode:

  * full AUROC using all features,
  * leave-one-out AUROC (drop each feature, RENORMALIZE the remaining weights
    to preserve their total mass) and the delta vs full,
  * single-signal-alone AUROC (each raw feature as the score by itself).

Capability only: it runs on the human's real CSV and prints/saves a table; it
fabricates nothing and writes no headline files unless --out is given.

Usage:
    python3 eval/ablation.py --csv eval/results_local_test.csv --mode local
    python3 eval/ablation.py --csv eval/results_api_test.csv --mode api \
        --weights eval/weights_fitted.json --out eval/ablation_api.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app import config_store  # noqa: E402
from backend.app.risk.hri_engine import (  # noqa: E402
    hri_from_features,
    FEATURE_NAMES_API,
    FEATURE_NAMES_LOCAL,
)

FEATURE_SETS = {"api": FEATURE_NAMES_API, "local": FEATURE_NAMES_LOCAL}


def loo_weights(block: Dict, drop_key: str) -> Dict:
    """Return a copy of ``block`` with ``drop_key`` removed and the remaining
    feature weights renormalized to preserve their original total mass."""
    feat_items = {
        k: v for k, v in block.items() if k != "sigmoid" and isinstance(v, (int, float))
    }
    orig_total = sum(feat_items.values())
    remaining = {k: v for k, v in feat_items.items() if k != drop_key}
    rem_total = sum(remaining.values())
    out: Dict = {}
    if rem_total != 0:
        factor = orig_total / rem_total
        for k, v in remaining.items():
            out[k] = v * factor
    else:
        out.update(remaining)
    if "sigmoid" in block:
        out["sigmoid"] = block["sigmoid"]
    return out


def _labels(rows: List[dict]) -> np.ndarray:
    return np.array([1 if r.get("model_match") == "mismatch" else 0 for r in rows])


def single_feature_auroc(rows: List[dict], feature: str) -> float:
    y, s = [], []
    for r in rows:
        v = r.get(feature, "")
        if r.get("model_match") not in ("match", "mismatch") or v in (None, ""):
            continue
        y.append(1 if r["model_match"] == "mismatch" else 0)
        s.append(float(v))
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def auroc_from_hri(
    rows: List[dict], feature_names: Sequence[str], weight_block: Dict, mode: str
) -> float:
    """AUROC of the HRI computed from ``weight_block`` over the rows."""
    y, s = [], []
    weights = {
        "sigmoid": weight_block.get("sigmoid", {"scale": 1.0, "bias": 0.0}),
        mode: weight_block,
    }
    for r in rows:
        if r.get("model_match") not in ("match", "mismatch"):
            continue
        feats, ok = {}, True
        for fn in feature_names:
            v = r.get(fn, "")
            if v in (None, ""):
                ok = False
                break
            feats[fn] = float(v)
        if not ok:
            continue
        y.append(1 if r["model_match"] == "mismatch" else 0)
        s.append(hri_from_features(feats, mode=mode, weights=weights))
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def run_ablation(rows: List[dict], mode: str, weight_block: Dict) -> List[dict]:
    feats = FEATURE_SETS[mode]
    full = auroc_from_hri(rows, feats, weight_block, mode)
    table = [{"config": "FULL", "auroc": full, "delta_vs_full": 0.0}]
    for drop in feats:
        block = loo_weights(weight_block, drop)
        auc = auroc_from_hri(rows, [f for f in feats if f != drop], block, mode)
        table.append({"config": f"-{drop}", "auroc": auc, "delta_vs_full": auc - full})
    for f in feats:
        table.append(
            {
                "config": f"only:{f}",
                "auroc": single_feature_auroc(rows, f),
                "delta_vs_full": single_feature_auroc(rows, f) - full,
            }
        )
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--mode", choices=list(FEATURE_SETS), default="local")
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Weights JSON (default: config/weights.json).",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.weights:
        weight_block = json.loads(args.weights.read_text())[args.mode]
    else:
        weight_block = config_store.weights()[args.mode]

    table = run_ablation(rows, args.mode, weight_block)
    print(f"\n=== Ablation ({args.mode}, n={len(rows)}) ===")
    print(f"  {'config':>22}  {'AUROC':>7}  {'delta':>7}")
    for t in table:
        print(f"  {t['config']:>22}  {t['auroc']:>7.3f}  {t['delta_vs_full']:>+7.3f}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config", "auroc", "delta_vs_full"])
            w.writeheader()
            w.writerows(table)
        print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
