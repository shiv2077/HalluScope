"""Re-score a feature CSV under a given weights file (cross-model transfer).

Transfer experiment: fit HRI weights on model A's TRAIN split
(eval/fit_weights.py --csv A_train --out weightsA.json), then apply those
weights to model B's TEST features here to produce a transferred ``hri``
column, and evaluate it on B with eval/analyze.py. This measures whether the
learned signal combination transfers across models.

Capability only: it writes a re-scored CSV (and optionally prints the
transferred AUROC + bootstrap CI, computed at YOUR runtime on real data). It
fabricates nothing.

Usage:
    # weightsA.json was produced by fit_weights on model A's train split
    python3 eval/score_features.py --csv eval/results_B_test.csv \
        --weights weightsA.json --mode local --out eval/results_B_transferA.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.risk.hri_engine import (  # noqa: E402
    hri_from_features, classify_risk,
    FEATURE_NAMES_API, FEATURE_NAMES_LOCAL,
)

FEATURE_SETS = {"api": FEATURE_NAMES_API, "local": FEATURE_NAMES_LOCAL}


def rescore_rows(rows: List[dict], mode: str, weights: Dict) -> List[dict]:
    """Return rows with ``hri`` / ``risk_level`` recomputed under ``weights``.

    Rows missing a required feature are left unchanged (hri set to None).
    """
    feats = FEATURE_SETS[mode]
    out: List[dict] = []
    for r in rows:
        r = dict(r)
        fvec, ok = {}, True
        for fn in feats:
            v = r.get(fn, "")
            if v in (None, ""):
                ok = False
                break
            fvec[fn] = float(v)
        if ok:
            hri = hri_from_features(fvec, mode=mode, weights=weights)
            r["hri"] = round(hri, 4)
            r["risk_level"] = classify_risk(hri)
        else:
            r["hri"] = None
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--mode", choices=list(FEATURE_SETS), default="local")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true",
                    help="Also print transferred AUROC + 95%% CI (needs model_match).")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    weights = json.loads(args.weights.read_text())
    rescored = rescore_rows(rows, args.mode, weights)

    fieldnames = list(rescored[0].keys()) if rescored else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rescored)
    print(f"Wrote {len(rescored)} re-scored rows -> {args.out}")

    if args.report:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from eval.stats import bootstrap_auroc_ci
        usable = [r for r in rescored
                  if r.get("hri") is not None and r.get("model_match") in ("match", "mismatch")]
        if len({r["model_match"] for r in usable}) == 2:
            y = np.array([1 if r["model_match"] == "mismatch" else 0 for r in usable])
            s = np.array([float(r["hri"]) for r in usable])
            lo, point, hi = bootstrap_auroc_ci(y, s)
            print(f"Transferred AUROC = {point:.3f}  (95% CI [{lo:.3f}, {hi:.3f}], n={len(usable)})")
        else:
            print("Cannot report AUROC: need both classes in model_match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
