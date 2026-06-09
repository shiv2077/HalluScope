"""Stage-3 PREVIEW: per-access-class LR detectors from the E2 cache (CPU).

Reads the strata records (qid, perturb_type, correct, n_correct_probes) and the
matching cached GenRecords, then for each access class (text/topk/dist/hidden)
and each knowledge subpopulation (knows = band high, unknown = band low) fits LR
on the CALIB split and scores VAL — AUROC of hallucination detection with a
1000-resample bootstrap CI. Test split is never touched.

Hypothesis being previewed: on the *knows* subpopulation, text/topk/dist hover
near 0.5 while hidden is well above; on *unknown*, all are above 0.5.

This is a PREVIEW, not E2. With the smoke cache the numbers are meaningless
(tiny N, likely NaN); the point is that the machinery runs end-to-end.

Run: .venv/bin/python scripts/run_s3_baselines.py --tag Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hv2.baselines import ACCESS_CLASSES, bootstrap_auroc, extract_features, fit_eval  # noqa: E402
from src.hv2.config import REPO_ROOT  # noqa: E402
from src.hv2.hdf5_store import H5Store  # noqa: E402
from src.hv2.probes import know_band  # noqa: E402
from src.hv2.splits import assign_split  # noqa: E402

SUBPOPS = {"knows": "high", "unknown": "low"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="model tag, e.g. Qwen2.5-1.5B-Instruct or *_smoke")
    args = ap.parse_args()

    cache = REPO_ROOT / "data" / "cache" / f"e2_{args.tag}.h5"
    recs_path = REPO_ROOT / "data" / f"strata_records_{args.tag}.jsonl"
    if not cache.exists() or not recs_path.exists():
        print(f"[s3] missing inputs (need {cache} and {recs_path}). Run E2 first.")
        return 2

    records = [json.loads(l) for l in recs_path.read_text().splitlines() if l.strip()]
    store = H5Store(cache, mode="r")

    # Build, per (access_class), feature matrix + label(=hallucination) + band + split.
    rows = []
    for r in records:
        key = f"{args.tag}/{r['qid']}/{r['perturb_type']}"
        if not store.has(key):
            continue
        rec = store.read(key)
        rows.append(
            {
                "rec": rec,
                "y": int(not r["correct"]),  # 1 = hallucination
                "band": know_band(r["n_correct_probes"]),
                "split": assign_split(r["qid"]),
            }
        )
    store.close()
    print(f"[s3] {len(rows)} cached generations loaded from {cache.name}")

    results = {}
    for ac in ACCESS_CLASSES:
        results[ac] = {}
        for sub, band in SUBPOPS.items():
            tr = [d for d in rows if d["band"] == band and d["split"] == "calib"]
            va = [d for d in rows if d["band"] == band and d["split"] == "val"]
            if len(tr) < 8 or len(va) < 8 or len({d["y"] for d in tr}) < 2 or len({d["y"] for d in va}) < 2:
                results[ac][sub] = {"auroc": float("nan"), "ci_low": float("nan"),
                                    "ci_high": float("nan"), "n": len(va), "note": "insufficient data"}
                continue
            Xtr = np.vstack([extract_features(d["rec"], ac) for d in tr])
            ytr = np.array([d["y"] for d in tr])
            Xva = np.vstack([extract_features(d["rec"], ac) for d in va])
            yva = np.array([d["y"] for d in va])
            scores = fit_eval(Xtr, ytr, Xva, yva)
            results[ac][sub] = bootstrap_auroc(yva, scores, n=1000)

    out_dir = REPO_ROOT / "results" / f"s3_preview_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "auroc.json").write_text(json.dumps(results, indent=2))

    # print a compact table
    print(f"[s3] AUROC (hallucination detection) — PREVIEW for tag={args.tag}")
    print(f"{'access':>8} | {'knows (KH/KC)':>22} | {'unknown (UH/UC)':>22}")
    for ac in ACCESS_CLASSES:
        def cell(sub):
            d = results[ac][sub]
            a = d["auroc"]
            return "n/a" if a != a else f"{a:.3f} [{d['ci_low']:.3f},{d['ci_high']:.3f}] n={d['n']}"
        print(f"{ac:>8} | {cell('knows'):>22} | {cell('unknown'):>22}")

    _try_plot(results, out_dir)
    print(f"[s3] wrote {out_dir/'auroc.json'}")
    return 0


def _try_plot(results, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(ACCESS_CLASSES))
        for i, (sub, _) in enumerate(SUBPOPS.items()):
            vals = [results[ac][sub]["auroc"] for ac in ACCESS_CLASSES]
            ax.bar(x + (i - 0.5) * 0.35, [0 if v != v else v for v in vals], width=0.35, label=sub)
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(ACCESS_CLASSES)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0, 1)
        ax.set_title("S3 preview: hallucination detection by access class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "auroc.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[s3] plot skipped: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
