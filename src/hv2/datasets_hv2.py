"""Sample the E1 question pool: 4000 TriviaQA + 2000 NQ-open (CLAUDE.md).

Uses HF ``datasets`` streaming with a seeded shuffle buffer so we pull only the
shards we need (disk-friendly) and the sample is reproducible. Each question is
normalized to ``{qid, question, gold_aliases, source}``. Gold aliases drive the
alias-EM grader.

Sampling method (documented for reproducibility): stream the validation split,
``.shuffle(seed, buffer_size=20000)``, take the first N. Fixed seed => fixed
sample. This is NOT a uniform draw over the whole split, but it is deterministic
and adequate for building a stratification pool.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

# id fallbacks in case the canonical name moved between datasets versions
_TQA_IDS = ["mandarjoshi/trivia_qa", "trivia_qa"]
_NQ_IDS = ["google-research-datasets/nq_open", "nq_open"]


def _load_stream(ids: list[str], config: str | None, split: str, seed: int, buffer_size: int):
    from datasets import load_dataset

    last_err = None
    for ds_id in ids:
        try:
            ds = load_dataset(ds_id, config, split=split, streaming=True) if config else load_dataset(
                ds_id, split=split, streaming=True
            )
            return ds.shuffle(seed=seed, buffer_size=buffer_size)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Could not load any of {ids}: {last_err}")


def sample_triviaqa(n: int, seed: int = 42, buffer_size: int = 20000) -> list[dict]:
    ds = _load_stream(_TQA_IDS, "rc.nocontext", "validation", seed, buffer_size)
    out = []
    for ex in itertools.islice(ds, n):
        ans = ex.get("answer", {}) or {}
        aliases = {ans.get("value", "")}
        aliases.update(ans.get("aliases", []) or [])
        aliases.update(ans.get("normalized_aliases", []) or [])
        aliases.discard("")
        qid = "tqa_" + str(ex.get("question_id") or hashlib.md5(ex["question"].encode()).hexdigest()[:12])
        out.append({"qid": qid, "question": ex["question"], "gold_aliases": sorted(aliases), "source": "triviaqa"})
    return out


def sample_nq_open(n: int, seed: int = 42, buffer_size: int = 20000) -> list[dict]:
    ds = _load_stream(_NQ_IDS, None, "validation", seed, buffer_size)
    out = []
    for ex in itertools.islice(ds, n):
        aliases = sorted({a for a in (ex.get("answer") or []) if a})
        qid = "nq_" + hashlib.md5(ex["question"].encode()).hexdigest()[:12]
        out.append({"qid": qid, "question": ex["question"], "gold_aliases": aliases, "source": "nq_open"})
    return out


def build_questions(
    n_triviaqa: int = 4000,
    n_nq_open: int = 2000,
    seed: int = 42,
    out_path: str | Path = "data/e1_questions.parquet",
) -> pd.DataFrame:
    """Sample both sources, de-dup by qid, persist parquet (+ sidecar json meta)."""
    rows = sample_triviaqa(n_triviaqa, seed) + sample_nq_open(n_nq_open, seed)
    df = pd.DataFrame(rows).drop_duplicates(subset="qid").reset_index(drop=True)
    # store aliases as JSON strings for parquet portability
    df["gold_aliases"] = df["gold_aliases"].apply(json.dumps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    meta = {
        "n_total": len(df),
        "n_triviaqa": int((df.source == "triviaqa").sum()),
        "n_nq_open": int((df.source == "nq_open").sum()),
        "seed": seed,
        "requested": {"triviaqa": n_triviaqa, "nq_open": n_nq_open},
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return df


def load_questions(path: str | Path = "data/e1_questions.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["gold_aliases"] = df["gold_aliases"].apply(json.loads)
    return df
