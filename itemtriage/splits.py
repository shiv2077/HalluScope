"""Subject-level calibration, validation and test splits for MMLU-Redux, stratified by base rate.

Procedure (also written verbatim into docs/PREREG.md):
1. Each subject's primary-class error rate is computed over its known items (category `expert` excluded).
   Subjects are sorted by that rate ascending, ties broken by subject name.
2. The sorted list is cut into consecutive blocks of 5, with a final shorter block (11 blocks of 5 and
   one of 2 for 57 subjects).
3. In each block of 5, 2 subjects go to calibration and 3 to test. In the final block of 2, 1 goes to each.
   numpy RandomState(42) chooses only which positions inside each block are calibration, block by block.
4. Validation inside calibration: the 23 calibration subjects are sorted the same way and cut into blocks
   of 4 with a final block of 3. One subject per block becomes validation, positions chosen by
   RandomState(43).
The frozen file is written once with a sha256 and refuses to change.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import labels as labels_mod
from . import paths

SPLIT_FILE = paths.SPLITS_DIR / "mmlu_redux.json"
PROCEDURE = "base-rate-stratified-blocks-v1"
TEST_BLOCK, CAL_PER_BLOCK, VAL_BLOCK = 5, 2, 4
SEED_TEST, SEED_VAL = 42, 43


def subject_rates(items: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    m = items[["item_id", "subject"]].merge(lab[["item_id", "flaw_primary"]], on="item_id")
    known = m.dropna(subset=["flaw_primary"])
    r = known.groupby("subject")["flaw_primary"].mean().rename("rate").reset_index()
    return r.sort_values(["rate", "subject"], kind="mergesort").reset_index(drop=True)


def _blocks(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def assign(sorted_subjects: list[str], block: int, take: int, seed: int, take_last: int = 1) -> tuple[list[str], list[str]]:
    """Pick `take` per full block (and `take_last` from a short final block). Returns (picked, rest)."""
    rng = np.random.RandomState(seed)
    picked, rest = [], []
    for b in _blocks(sorted_subjects, block):
        k = take if len(b) == block else min(take_last, len(b))
        pos = set(rng.choice(len(b), size=k, replace=False).tolist())
        for i, s in enumerate(b):
            (picked if i in pos else rest).append(s)
    return picked, rest


def derive(items: pd.DataFrame, lab: pd.DataFrame) -> dict:
    rates = subject_rates(items, lab)
    order = rates["subject"].tolist()
    cal, test = assign(order, TEST_BLOCK, CAL_PER_BLOCK, SEED_TEST)
    cal_sorted = [s for s in order if s in set(cal)]
    val, fit = assign(cal_sorted, VAL_BLOCK, 1, SEED_VAL)
    subjects = {"calibration_fit": sorted(fit), "validation": sorted(val), "test": sorted(test)}
    by_subject = items.groupby("subject")["item_id"].apply(sorted).to_dict()
    item_ids = {k: sorted(i for s in v for i in by_subject[s]) for k, v in subjects.items()}
    body = {"dataset": "mmlu_redux_2.0", "procedure": PROCEDURE, "seeds": {"test": SEED_TEST, "validation": SEED_VAL},
            "subjects": subjects, "item_ids": item_ids}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def check_disjoint(split: dict, items: pd.DataFrame) -> None:
    folds = list(split["item_ids"])
    ids = {k: set(v) for k, v in split["item_ids"].items()}
    th = items.set_index("item_id")["text_hash"]
    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            if ids[a] & ids[b]:
                raise AssertionError(f"{a} and {b} share item ids")
            shared = set(th[list(ids[a])]) & set(th[list(ids[b])])
            if shared:
                raise AssertionError(f"{a} and {b} share {len(shared)} question texts, e.g. {sorted(shared)[:3]}")


def freeze() -> dict:
    items = pd.read_parquet(paths.ITEMS)
    lab = labels_mod.load(caller="itemtriage.splits")
    split = derive(items, lab)
    check_disjoint(split, items)
    if SPLIT_FILE.exists():
        old = json.loads(SPLIT_FILE.read_text())
        if old["sha256"] != split["sha256"]:
            raise RuntimeError(f"refusing to change frozen split {SPLIT_FILE}: {old['sha256']} != {split['sha256']}")
        return old
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1, sort_keys=True))
    return split


def load() -> dict:
    s = json.loads(SPLIT_FILE.read_text())
    body = {k: v for k, v in s.items() if k != "sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != s["sha256"]:
        raise RuntimeError(f"{SPLIT_FILE} hash mismatch; the split file was edited")
    return s


def fold_of(split: dict) -> dict[str, str]:
    return {i: k for k, v in split["item_ids"].items() for i in v}
