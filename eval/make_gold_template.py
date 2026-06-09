"""Build a GOLD answer-equivalence labelling template for grader calibration.

Samples N (default 80) DISJOINT HalluEval-QA examples (fixed seed, reusing
build_subset's dedupe + seed logic), excluding any question already used in the
train/test/legacy split files so the calibration set never overlaps the eval
sets. Emits, under eval/gold/:

    gold_questions.json     [{id, question, reference_answer}]  -> feed to
                            run_benchmark.py to generate REAL model answers.
    gold_annotatorA.csv     id, question, reference_answer, model_answer(BLANK),
    gold_annotatorB.csv     human_label(BLANK)   -- two identical copies for
                            two-annotator inter-rater agreement.
    README_GOLD.txt         step-by-step instructions.

IMPORTANT: model_answer is left BLANK. It must come from a REAL generation the
human runs (run_benchmark on gold_questions.json), then merged back with
``--fill-from``. This tool NEVER invents model answers.

Usage:
    # 1) create the template
    python3 eval/make_gold_template.py --n 80

    # 2) (human) generate real answers for the gold questions, e.g. local model:
    #    python3 eval/run_benchmark.py --mode local --eval-deterministic \
    #        --dataset-file eval/gold/gold_questions.json \
    #        --out eval/gold/gold_results.csv
    #    python3 eval/match.py --csv eval/gold/gold_results.csv   # optional

    # 3) merge the real model answers into the annotator CSVs:
    python3 eval/make_gold_template.py --fill-from eval/gold/gold_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.build_subset import (  # noqa: E402
    SEED,
    DATASETS_DIR,
    _dedupe_unique,
    _load_hf,
)

GOLD_DIR = ROOT / "eval" / "gold"
QUESTIONS_PATH = GOLD_DIR / "gold_questions.json"
ANNOTATOR_PATHS = [GOLD_DIR / "gold_annotatorA.csv", GOLD_DIR / "gold_annotatorB.csv"]
GOLD_COLUMNS = ["id", "question", "reference_answer", "model_answer", "human_label"]

# Existing split / legacy files whose questions must be EXCLUDED from gold.
EXCLUDE_FILES = [
    DATASETS_DIR / "halueval_qa_train.json",
    DATASETS_DIR / "halueval_qa_test.json",
    DATASETS_DIR / "halluveval_qa_questions_100.json",
]


def _excluded_questions() -> Set[str]:
    out: Set[str] = set()
    for p in EXCLUDE_FILES:
        if p.exists():
            for row in json.loads(p.read_text()):
                q = (row.get("question") or "").strip()
                if q:
                    out.add(q)
    return out


def select_gold(
    pool_rows: List[dict],
    n: int,
    seed: int,
    exclude_questions: Set[str],
    dataset: str = "halueval_qa",
) -> List[dict]:
    """Deterministically pick N unique-question rows DISJOINT from the excluded
    set. Returns [{id, question, reference_answer}]."""
    unique = _dedupe_unique(pool_rows, dataset)
    eligible = [r for r in unique if r["question"] not in exclude_questions]
    order = list(range(len(eligible)))
    random.Random(seed).shuffle(order)
    if len(eligible) < n:
        raise RuntimeError(
            f"Only {len(eligible)} eligible unique questions after exclusion; "
            f"need {n}."
        )
    picked = [eligible[i] for i in order[:n]]
    return [
        {
            "id": f"gold_{i:05d}",
            "question": r["question"],
            "reference_answer": r["reference_answer"],
        }
        for i, r in enumerate(picked, 1)
    ]


def fill_model_answers(
    annotator_rows: List[dict], answers_by_id: Dict[str, str]
) -> int:
    """Fill BLANK model_answer cells from real generations keyed by id.

    Never overwrites a non-empty cell; never invents an answer (ids absent from
    answers_by_id stay blank). Returns the number of cells filled.
    """
    filled = 0
    for r in annotator_rows:
        if (r.get("model_answer") or "").strip():
            continue
        ans = answers_by_id.get(r["id"], "")
        if ans:
            r["model_answer"] = ans
            filled += 1
    return filled


# ── IO helpers ───────────────────────────────────────────────────────
def _write_template(rows: List[dict]) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for path in ANNOTATOR_PATHS:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=GOLD_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "id": r["id"],
                        "question": r["question"],
                        "reference_answer": r["reference_answer"],
                        "model_answer": "",
                        "human_label": "",
                    }
                )
    (GOLD_DIR / "README_GOLD.txt").write_text(_README)


def _read_results_answers(path: Path) -> Dict[str, str]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows or "id" not in rows[0] or "model_answer" not in rows[0]:
        raise SystemExit(
            f"{path} must have 'id' and 'model_answer' columns "
            "(a real run_benchmark output)."
        )
    return {r["id"]: (r.get("model_answer") or "") for r in rows}


_README = """GOLD GRADER-CALIBRATION WORKFLOW
================================
1. (done) Template created: gold_questions.json + gold_annotatorA.csv +
   gold_annotatorB.csv. model_answer is intentionally BLANK.

2. Generate the model's REAL answers for these exact questions:
     python3 eval/run_benchmark.py --mode local --eval-deterministic \\
         --dataset-file eval/gold/gold_questions.json \\
         --out eval/gold/gold_results.csv
   (use --mode api for the API model; needs OPENAI_API_KEY)

3. Merge those real answers into BOTH annotator CSVs:
     python3 eval/make_gold_template.py --fill-from eval/gold/gold_results.csv

4. Hand-label: in gold_annotatorA.csv and gold_annotatorB.csv set human_label
   to 'correct' or 'incorrect' for every row, judging whether model_answer is
   equivalent to reference_answer. Two people label independently.

5. Calibrate against the human labels:
     python3 eval/grader_calibration.py \\
         --gold eval/gold/gold_annotatorA.csv \\
         --gold-b eval/gold/gold_annotatorB.csv

6. Review inter-annotator kappa, then set config/grader.json -> sbert_threshold
   to the recommended value (the command is printed by step 5).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="Gold set size (default 80).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--fill-from",
        type=Path,
        default=None,
        help="Merge REAL model answers from a run_benchmark output "
        "CSV into the annotator templates (by id).",
    )
    args = ap.parse_args()

    if args.fill_from is not None:
        if not all(p.exists() for p in ANNOTATOR_PATHS):
            raise SystemExit(
                "Annotator templates not found; run without "
                "--fill-from first to create them."
            )
        answers = _read_results_answers(args.fill_from)
        for path in ANNOTATOR_PATHS:
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            n_filled = fill_model_answers(rows, answers)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=GOLD_COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"Filled {n_filled} model_answer cells in {path.name}")
        print(
            "Now hand-label human_label in both files, then run "
            "eval/grader_calibration.py."
        )
        return 0

    pool = _load_hf("halueval_qa")
    rows = select_gold(pool, args.n, args.seed, _excluded_questions())
    _write_template(rows)
    print(f"Wrote gold template ({len(rows)} rows) to {GOLD_DIR}/")
    print(f"  - {QUESTIONS_PATH.name} (feed to run_benchmark)")
    print(
        f"  - {ANNOTATOR_PATHS[0].name}, {ANNOTATOR_PATHS[1].name} "
        "(model_answer BLANK -> fill with REAL answers, then hand-label)"
    )
    print("  - README_GOLD.txt (full workflow)")
    print(
        "\nmodel_answer is intentionally empty. Do NOT invent answers — "
        "generate them with run_benchmark, then --fill-from."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
