"""Build the v2 eval subset for HalluScope's HalluEval QA benchmark.

NEW METHODOLOGY (v2):
The original v1 subset paired each question with EITHER a right or
hallucinated answer. That introduced a length artifact: 'correct'
candidates were short fragments and 'hallucinated' candidates were
full sentences, so HRI (computed on the LLM's evaluation reasoning)
ended up tracking input fluency rather than truth.

In v2 we drop the candidate answer entirely. Each row is just a unique
question + the dataset's reference right_answer. The benchmark runner
will pass only the question to HalluScope, capture HRI on the LLM's
own free-form answer, and use sentence-transformers cosine similarity
between (model answer, reference answer) to derive the ground truth
match/mismatch label.

Output: eval/datasets/halluveval_qa_questions_100.json
Schema per row:
    {"id": "halu_q_0001", "question": str, "reference_answer": str}

Run from repo root:
    python3 eval/build_subset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "datasets" / "halluveval_qa_questions_100.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DATASET_ID = "pminervini/HaluEval"
SUBSET = "qa"
N_QUESTIONS = 100
SEED = 42


def main() -> None:
    from datasets import load_dataset

    print(f"Loading {DATASET_ID} / {SUBSET} ...")
    ds = load_dataset(DATASET_ID, SUBSET)
    split_name = next(iter(ds.keys()))
    rows = list(ds[split_name])
    print(f"  loaded {len(rows)} rows from split '{split_name}'")

    random.seed(SEED)
    indices = list(range(len(rows)))
    random.shuffle(indices)

    seen_questions: set[str] = set()
    out: list[dict] = []
    for idx in indices:
        if len(out) >= N_QUESTIONS:
            break
        row = rows[idx]
        q = (row.get("question") or row.get("input") or "").strip()
        ref = (row.get("right_answer") or row.get("answer") or "").strip()
        if not q or not ref or q in seen_questions:
            continue
        seen_questions.add(q)
        out.append({
            "id": f"halu_q_{len(out) + 1:04d}",
            "question": q,
            "reference_answer": ref,
        })

    if len(out) < N_QUESTIONS:
        raise RuntimeError(
            f"Could not collect {N_QUESTIONS} unique questions; got {len(out)}"
        )

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} unique questions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
