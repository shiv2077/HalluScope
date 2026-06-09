"""Build deterministic eval subsets / splits for HalluScope.

v2 methodology: each row is a unique question + the dataset's reference answer.
The benchmark passes ONLY the question to HalluScope and grades the model's
own free-form answer against the reference (eval/match.py).

This script supports two datasets and two output modes:

  Datasets (--dataset):
    halueval_qa  (pminervini/HaluEval, 'qa')   [default]
    truthfulqa   (truthful_qa, 'generation')   -- claimed in the paper; the
                 loader + grading path exist here, but NO numbers are produced.

  Modes:
    (default)        single subset -> eval/datasets/<name>_questions_100.json
                     (back-compat with eval/REPORT.md's documented command).
    --split          DISJOINT train/test -> eval/datasets/<name>_train.json
                     and <name>_test.json, sized by --n-train / --n-test.

Determinism: a single seeded shuffle drives both splits; questions are
deduplicated and train/test never share a question.

Run from repo root:
    python3 eval/build_subset.py                                   # 100 HalluEval QA
    python3 eval/build_subset.py --split --n-train 1000 --n-test 1000
    python3 eval/build_subset.py --dataset truthfulqa --split \
        --n-train 400 --n-test 400
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

DATASET_SPECS = {
    "halueval_qa": {"hf_id": "pminervini/HaluEval", "config": "qa"},
    "truthfulqa": {"hf_id": "truthful_qa", "config": "generation"},
}

SEED = 42


def normalize_row(row: dict, dataset: str) -> dict:
    """Map a raw HF row onto the common schema {question, reference_answer, ...}."""
    if dataset == "truthfulqa":
        return {
            "question": (row.get("question") or "").strip(),
            "reference_answer": (row.get("best_answer") or "").strip(),
            "correct_answers": list(row.get("correct_answers") or []),
            "incorrect_answers": list(row.get("incorrect_answers") or []),
        }
    # halueval_qa
    return {
        "question": (row.get("question") or row.get("input") or "").strip(),
        "reference_answer": (
            row.get("right_answer") or row.get("answer") or ""
        ).strip(),
    }


def _dedupe_unique(rows: List[dict], dataset: str) -> List[dict]:
    """Deterministically dedupe rows by question (input order preserved)."""
    seen: set = set()
    out: List[dict] = []
    for r in rows:
        nr = normalize_row(r, dataset)
        q = nr["question"]
        if not q or not nr["reference_answer"] or q in seen:
            continue
        seen.add(q)
        out.append(nr)
    return out


def build_split(
    rows: List[dict],
    n_train: int,
    n_test: int,
    seed: int = SEED,
    dataset: str = "halueval_qa",
) -> Tuple[List[dict], List[dict]]:
    """Return disjoint (train, test) lists of unique-question rows.

    A single seeded shuffle orders the deduped rows; the first ``n_train`` go
    to train and the next ``n_test`` to test, so the two never overlap. IDs are
    assigned per split.
    """
    unique = _dedupe_unique(rows, dataset)
    order = list(range(len(unique)))
    random.Random(seed).shuffle(order)
    shuffled = [unique[i] for i in order]

    if len(shuffled) < n_train + n_test:
        raise RuntimeError(
            f"Need {n_train + n_test} unique questions; only {len(shuffled)} available."
        )

    def _tag(items: List[dict], prefix: str) -> List[dict]:
        tagged = []
        for i, it in enumerate(items, 1):
            tagged.append({"id": f"{prefix}_{i:05d}", **it})
        return tagged

    train = _tag(shuffled[:n_train], "train")
    test = _tag(shuffled[n_train : n_train + n_test], "test")
    return train, test


def _legacy_subset(rows: List[dict], n: int, seed: int, dataset: str) -> List[dict]:
    """Reproduce the ORIGINAL committed subset byte-for-byte (shuffle raw
    indices, then dedupe), so re-running the default does not silently change
    the dataset that existing eval/results_*.csv are paired with.
    """
    random.seed(seed)
    indices = list(range(len(rows)))
    random.shuffle(indices)
    seen: set = set()
    out: List[dict] = []
    for idx in indices:
        if len(out) >= n:
            break
        nr = normalize_row(rows[idx], dataset)
        q, ref = nr["question"], nr["reference_answer"]
        if not q or not ref or q in seen:
            continue
        seen.add(q)
        out.append({"id": f"halu_q_{len(out) + 1:04d}", **nr})
    if len(out) < n:
        raise RuntimeError(f"Could not collect {n} unique questions; got {len(out)}")
    return out


def _load_hf(dataset: str) -> List[dict]:
    from datasets import load_dataset

    spec = DATASET_SPECS[dataset]
    print(f"Loading {spec['hf_id']} / {spec['config']} ...")
    ds = load_dataset(spec["hf_id"], spec["config"])
    split_name = next(iter(ds.keys()))
    rows = list(ds[split_name])
    print(f"  loaded {len(rows)} rows from split '{split_name}'")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_SPECS), default="halueval_qa")
    ap.add_argument(
        "--split",
        action="store_true",
        help="Emit disjoint train.json + test.json instead of one subset.",
    )
    ap.add_argument("--n", type=int, default=100, help="Subset size (non-split mode).")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_hf(args.dataset)

    if args.split:
        train, test = build_split(
            rows, args.n_train, args.n_test, seed=args.seed, dataset=args.dataset
        )
        train_path = DATASETS_DIR / f"{args.dataset}_train.json"
        test_path = DATASETS_DIR / f"{args.dataset}_test.json"
        train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False))
        test_path.write_text(json.dumps(test, indent=2, ensure_ascii=False))
        print(f"Wrote {len(train)} -> {train_path}")
        print(f"Wrote {len(test)}  -> {test_path}")
    else:
        # Legacy single-subset path (byte-identical to the committed file).
        subset = _legacy_subset(rows, args.n, args.seed, args.dataset)
        name = (
            "halluveval_qa_questions_100"
            if args.dataset == "halueval_qa" and args.n == 100
            else f"{args.dataset}_questions_{args.n}"
        )
        out_path = DATASETS_DIR / f"{name}.json"
        out_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False))
        print(f"Wrote {len(subset)} unique questions -> {out_path}")


if __name__ == "__main__":
    main()
