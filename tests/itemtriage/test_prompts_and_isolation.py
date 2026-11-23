"""Prompt leak guard, label isolation, and the frozen split."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from itemtriage import labels, paths, prompts, splits  # noqa: E402

needs_data = pytest.mark.skipif(not paths.ITEMS.exists(), reason="run itemtriage ingest first")
LABEL_FIELDS = {"error_type", "error_type_raw", "correct_answer", "correct_answer_raw", "potential_reason",
                "flaw_primary", "flaw_any"}


@needs_data
def test_wrong_key_item_passes_and_injected_annotation_fails():
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")
    lab = labels.load(caller="itemtriage.audit").set_index("item_id")
    wrong = lab.index[lab["error_type_raw"] == "wrong_groundtruth"]
    it = items.loc[wrong[0]].to_dict() | {"item_id": wrong[0]}
    reason = lab.loc[wrong[0], "potential_reason"] or "The keyed answer contradicts the cited source text."
    text, order = prompts.mc_prompt(it, shift=1)
    prompts.check_prompt(text, it, order, forbidden_texts=(reason,))  # options include the correct one: allowed
    with pytest.raises(AssertionError):
        prompts.check_prompt(text + "\nNote: " + reason + " (annotator)", it, order, forbidden_texts=(reason,))
    with pytest.raises(AssertionError):
        prompts.check_prompt(text + "\nwrong_groundtruth", it, order)


def test_plain_words_are_not_category_hits():
    it = {"item_id": "x", "subject": "law", "question": "Is it ok to call an expert witness?", "options": ["Yes", "No"],
          "key_index": 0}
    text, order = prompts.mc_prompt(it)
    prompts.check_prompt(text, it, order)


def test_tampered_options_fail():
    it = {"item_id": "x", "subject": "law", "question": "q?", "options": ["a\nb", "c", "d", "e"], "key_index": 0}
    text, order = prompts.mc_prompt(it, shift=1)
    prompts.check_prompt(text, it, order)
    with pytest.raises(AssertionError):
        prompts.check_prompt(text.replace("\nA. c\n", "\nA. z\n"), it, order)


@needs_data
def test_items_table_has_no_label_columns():
    assert not (set(pd.read_parquet(paths.ITEMS).columns) & LABEL_FIELDS)


def test_label_loader_refuses_unlisted_callers():
    with pytest.raises(PermissionError):
        labels.load(caller="itemtriage.scorers")


def test_label_free_modules_never_touch_labels():
    for mod in ("prompts", "panel_api", "panel_local", "scorers", "irt", "rank", "hidden"):
        p = ROOT / "itemtriage" / f"{mod}.py"
        if not p.exists():
            continue
        tree = ast.parse(p.read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "labels" not in names and not any(m.endswith("labels") for m in mods), mod
        assert "labels.parquet" not in p.read_text() and "paths.LABELS" not in p.read_text(), mod


@needs_data
def test_split_is_rederivable_frozen_and_disjoint():
    items = pd.read_parquet(paths.ITEMS)
    frozen = splits.load()
    again = splits.derive(items, labels.load(caller="itemtriage.splits"))
    assert again["sha256"] == frozen["sha256"]
    splits.check_disjoint(frozen, items)
    sizes = {k: len(v) for k, v in frozen["subjects"].items()}
    assert sizes == {"calibration_fit": 17, "validation": 6, "test": 34}
