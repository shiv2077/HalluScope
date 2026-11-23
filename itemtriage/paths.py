"""Paths and fixed vocabularies for itemtriage. Directories can be redirected for tests."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ITEMTRIAGE_DATA_DIR", ROOT / "data" / "itemtriage"))
RESULTS = Path(os.environ.get("ITEMTRIAGE_RESULTS_DIR", ROOT / "results" / "itemtriage"))
FIGURES = Path(os.environ.get("ITEMTRIAGE_FIGURES_DIR", ROOT / "figures" / "itemtriage"))
REPORTS = Path(os.environ.get("ITEMTRIAGE_REPORTS_DIR", ROOT / "reports"))
ROUTING_CALLS = ROOT / "results" / "calls.jsonl"

ITEMS = DATA / "items.parquet"
LABELS = DATA / "labels.parquet"
SPLITS_DIR = DATA / "splits"
RAW = DATA / "raw"

# MMLU-Redux 2.0 error_type identifiers, exactly as they appear in the dataset.
OK = "ok"
PRIMARY = ("wrong_groundtruth", "no_correct_answer", "multiple_correct_answers")
SECONDARY_ONLY = ("bad_question_clarity", "bad_options_clarity")
UNKNOWN = ("expert",)  # the annotator could not verify the item; excluded from both classes
CATEGORY_IDS = (OK, *PRIMARY, *SECONDARY_ONLY, *UNKNOWN)

LETTERS = "ABCDEFGHIJ"
