"""SBERT-based answer matcher for the HalluScope benchmark.

Given the dataset's reference_answer and the LLM's free-form
model_answer, decide whether they convey the same fact.

Approach: encode both with the model named in config/grader.json, take
cosine similarity, and threshold at config/grader.json -> sbert_threshold
(single source of truth; default 0.60, the original spec value). The
threshold is consequential — it defines the positive (hallucination) class —
so before reporting numbers run eval/grader_sensitivity.py to justify it.
This script also prints the cosine distribution after a run.

Standalone:
    # Run sanity tests
    python3 eval/match.py --selftest

    # Fill in model_match column for an existing results CSV
    python3 eval/match.py --csv eval/results_api.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app import config_store  # noqa: E402

# Single source of truth: config/grader.json. CLI --threshold still overrides.
_GRADER = config_store.grader()
DEFAULT_THRESHOLD = float(_GRADER["sbert_threshold"])
MODEL_NAME = _GRADER["sbert_model"]

_model = None  # lazy singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def cosine_similarity(a: str, b: str) -> float:
    """Single-pair cosine similarity. Convenience wrapper around batch_cosine."""
    sims = batch_cosine([a], [b])
    return float(sims[0])


def batch_cosine(a: Sequence[str], b: Sequence[str]) -> List[float]:
    """Vectorised pairwise cosine — preferred for >1 pair."""
    model = _get_model()
    emb_a = model.encode(list(a), convert_to_numpy=True, normalize_embeddings=True)
    emb_b = model.encode(list(b), convert_to_numpy=True, normalize_embeddings=True)
    sims = (emb_a * emb_b).sum(axis=1)  # already L2-normalised → dot = cosine
    return [float(s) for s in sims]


def classify_match(
    model_answer: str,
    reference_answer: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Return 'match' if cos(model_answer, reference_answer) >= threshold."""
    sim = cosine_similarity(model_answer, reference_answer)
    return "match" if sim >= threshold else "mismatch"


def fill_csv(
    csv_path: Path, threshold: float = DEFAULT_THRESHOLD
) -> Tuple[int, int, List[float]]:
    """Read csv_path, add model_match for every row, write in-place.
    Returns (n_match, n_mismatch, all_sims)."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"{csv_path} is empty")
    if "model_answer" not in rows[0] or "reference_answer" not in rows[0]:
        raise RuntimeError(
            f"{csv_path} is missing model_answer/reference_answer columns; "
            "are you running this on a v1 results file?"
        )

    sims = batch_cosine(
        [r["model_answer"] for r in rows],
        [r["reference_answer"] for r in rows],
    )

    n_match = n_mismatch = 0
    for r, s in zip(rows, sims):
        r["model_match"] = "match" if s >= threshold else "mismatch"
        r["sbert_cos"] = f"{s:.4f}"
        if r["model_match"] == "match":
            n_match += 1
        else:
            n_mismatch += 1

    # Preserve original column order, append sbert_cos at the end
    fieldnames = list(rows[0].keys())
    if "sbert_cos" not in fieldnames:
        fieldnames.append("sbert_cos")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    return n_match, n_mismatch, sims


def _print_distribution(sims: List[float], threshold: float) -> None:
    print()
    print(
        f"=== Cosine similarity distribution (n={len(sims)}, threshold={threshold}) ==="
    )
    print(
        f"  mean={statistics.mean(sims):.3f}  median={statistics.median(sims):.3f}  "
        f"std={statistics.stdev(sims):.3f}  min={min(sims):.3f}  max={max(sims):.3f}"
    )
    bins = [0] * 10
    for s in sims:
        # Map [-1, 1] -> [0, 10), but we expect cos to be in [0, 1] mostly
        idx = max(0, min(int(s * 10), 9))
        bins[idx] += 1
    for i, b in enumerate(bins):
        marker = " <- threshold" if (i / 10 <= threshold < (i + 1) / 10) else ""
        print(f"  [{i/10:.1f}-{(i+1)/10:.1f}) {'#' * b} ({b}){marker}")


def selftest() -> None:
    print("--- Sanity test 1: paraphrase that should match ---")
    s1 = cosine_similarity("Paris", "The capital of France is Paris.")
    label1 = classify_match("Paris", "The capital of France is Paris.")
    print(f"  cos={s1:.4f}  -> {label1}")
    assert label1 == "match", f"expected match, got {label1} (cos={s1})"

    print("--- Sanity test 2: unrelated facts that should mismatch ---")
    s2 = cosine_similarity("Paris", "Madrid is in Spain.")
    label2 = classify_match("Paris", "Madrid is in Spain.")
    print(f"  cos={s2:.4f}  -> {label2}")
    assert label2 == "mismatch", f"expected mismatch, got {label2} (cos={s2})"

    print()
    print("OK: both sanity tests passed.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Print details for the first N rows (default 0).",
    )
    args = ap.parse_args()

    if args.selftest:
        selftest()

    if args.csv:
        if not args.csv.exists():
            print(f"ERROR: {args.csv} not found", file=sys.stderr)
            return 2
        print(f"Filling model_match in {args.csv} (threshold={args.threshold})")
        n_match, n_mismatch, sims = fill_csv(args.csv, args.threshold)
        print(f"  match    : {n_match}")
        print(f"  mismatch : {n_mismatch}")
        _print_distribution(sims, args.threshold)

        if args.sample:
            print()
            print(f"=== First {args.sample} rows ===")
            with open(args.csv, encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= args.sample:
                        break
                    print(
                        f"  [{row['id']}] cos={row['sbert_cos']}  "
                        f"{row['model_match']:>8}  "
                        f"ref={row['reference_answer'][:40]!r}  "
                        f"model={row['model_answer'][:40]!r}"
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
