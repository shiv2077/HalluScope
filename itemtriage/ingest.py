"""Ingest MMLU-Redux 2.0 into a label-free item table and a separate label table.

items.parquet holds only what a model may see: question, options, key. labels.parquet holds the
annotation fields (error_type, correct_answer, potential_reason) and is read only via itemtriage.labels.
The raw download is cached under data/itemtriage/raw so re-ingest needs no network.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

from . import paths

REDUX_REPO = "edinburgh-dawg/mmlu-redux-2.0"
RAW_REDUX = paths.RAW / "mmlu_redux_2.0.parquet"
ITEM_COLS = ["item_id", "source", "subject", "question", "options", "n_options", "key_index", "text_hash"]
LABEL_COLS = ["item_id", "error_type_raw", "flaw_primary", "flaw_any", "correct_answer_raw", "potential_reason",
              "label_source"]


def normalize(text: str) -> str:
    """NFKC, lowercase, collapsed whitespace. Punctuation is kept: "J >> a" and "J << a" differ."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", t).strip()


def text_hash(question: str, options: list) -> str:
    """Identity of an item: question plus options. Generic stems such as "Which of the following is
    true?" recur with different options and are different items."""
    blob = normalize(question) + "||" + "||".join(normalize(o) for o in options)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def download_redux() -> pd.DataFrame:
    """Fetch all 57 subject configs once and cache them. About 5,700 rows, a few MB."""
    if RAW_REDUX.exists():
        return pd.read_parquet(RAW_REDUX)
    from datasets import load_dataset
    from huggingface_hub import list_repo_files

    subjects = sorted({f.split("/")[0] for f in list_repo_files(REDUX_REPO, repo_type="dataset") if "/" in f})
    frames = []
    for s in subjects:
        df = load_dataset(REDUX_REPO, s, split="test").to_pandas()
        df.insert(0, "row_in_subject", range(len(df)))
        df.insert(0, "subject", s)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    RAW_REDUX.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_REDUX, index=False)
    return raw


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["item_id"] = [f"redux:{s}:{i:03d}" for s, i in zip(raw["subject"], raw["row_in_subject"])]
    unknown_types = set(raw["error_type"]) - set(paths.CATEGORY_IDS)
    if unknown_types:
        raise ValueError(f"unexpected error_type values {sorted(unknown_types)}; update paths.CATEGORY_IDS")
    items = pd.DataFrame({
        "item_id": raw["item_id"],
        "source": "mmlu_redux_2.0",
        "subject": raw["subject"],
        "question": raw["question"],
        "options": raw["choices"].map(list),
        "n_options": raw["choices"].map(len),
        "key_index": raw["answer"].astype(int),
        "text_hash": [text_hash(q, list(o)) for q, o in zip(raw["question"], raw["choices"])],
    })[ITEM_COLS]
    et = raw["error_type"]
    unknown = et.isin(paths.UNKNOWN)
    labels = pd.DataFrame({
        "item_id": raw["item_id"],
        "error_type_raw": et,
        "flaw_primary": et.isin(paths.PRIMARY).astype("Int64").where(~unknown),
        "flaw_any": (et != paths.OK).astype("Int64").where(~unknown),
        "correct_answer_raw": raw["correct_answer"].fillna(""),
        "potential_reason": raw["potential_reason"].fillna(""),
        "label_source": "redux_expert",
    })[LABEL_COLS]
    return items, labels


def run() -> dict:
    raw = download_redux()
    items, labels = build(raw)
    paths.DATA.mkdir(parents=True, exist_ok=True)
    items.to_parquet(paths.ITEMS, index=False)
    labels.to_parquet(paths.LABELS, index=False)
    dup = items["text_hash"].duplicated(keep=False)
    return {"n_items": len(items), "n_subjects": int(items["subject"].nunique()),
            "error_type_counts": labels["error_type_raw"].value_counts().to_dict(),
            "n_primary": int(labels["flaw_primary"].sum()), "n_any": int(labels["flaw_any"].sum()),
            "n_unknown": int(labels["flaw_primary"].isna().sum()),
            "duplicate_question_items": int(dup.sum()),
            "duplicate_question_groups": int(items.loc[dup, "text_hash"].nunique())}
