"""Data contract for the routing layer.

Four CSVs per split under data/:
  <split>_questions.csv    qid, question, reference_answer
  <split>_generations.csv  qid, model, role, answer_text, token_logprobs (JSON), input_tokens,
                           output_tokens, temperature, seed
  <split>_labels.csv       qid, model, judge, verdict, reason
  <split>_scores.csv       qid, model, <one column per scorer>, extra_calls, extra_cost_usd

Calibration and test are different files with disjoint qids. load_split asserts that.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ROUTING_DATA_DIR", ROOT / "data"))
RESULTS = Path(os.environ.get("ROUTING_RESULTS_DIR", ROOT / "results"))
FIGURES = Path(os.environ.get("ROUTING_FIGURES_DIR", ROOT / "figures"))
REPORTS = Path(os.environ.get("ROUTING_REPORTS_DIR", ROOT / "reports"))
CONFIG = ROOT / "config"

SPLITS = ("calibration", "test")
ROLES = ("primary", "resample_1", "resample_2", "resample_3", "strong", "ptrue")
JUDGES = ("judge_a", "judge_b")
VERDICTS = ("CORRECT", "INCORRECT", "AMBIGUOUS")
SCORERS = ("mean_logprob", "min_k_logprob", "length", "length_residual", "consistency_k3", "p_true")

Role = Literal["primary", "resample_1", "resample_2", "resample_3", "strong", "ptrue"]
Judge = Literal["judge_a", "judge_b"]
Verdict = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]


class Question(BaseModel):
    qid: str
    question: str
    reference_answer: str


class Generation(BaseModel):
    qid: str
    model: str
    role: Role
    answer_text: str
    token_logprobs: str  # JSON list
    input_tokens: int
    output_tokens: int
    temperature: float
    seed: int


class Label(BaseModel):
    qid: str
    model: str
    judge: Judge
    verdict: Verdict
    reason: str


def models_config() -> dict:
    return json.loads((CONFIG / "models.json").read_text())


def path(split: str, kind: str) -> Path:
    assert split in SPLITS, split
    return DATA / f"{split}_{kind}.csv"


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _read(p: Path, model_cls) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist. Run `python -m routing.cli run-all` (needs API keys).")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    try:
        for rec in df.to_dict("records"):
            model_cls(**rec)
    except ValidationError as e:
        raise ValueError(f"{p} violates the data contract: {e}") from e
    return df


def read_questions(split: str) -> pd.DataFrame:
    return _read(path(split, "questions"), Question)


def read_generations(split: str) -> pd.DataFrame:
    df = _read(path(split, "generations"), Generation)
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)
    df["temperature"] = df["temperature"].astype(float)
    df["seed"] = df["seed"].astype(int)
    return df


def read_labels(split: str) -> pd.DataFrame:
    return _read(path(split, "labels"), Label)


def read_scores(split: str) -> pd.DataFrame:
    p = path(split, "scores")
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist. Run `python -m routing.cli score --split {split}`.")
    df = pd.read_csv(p)
    for s in SCORERS:
        if s not in df.columns:
            raise ValueError(f"{p} is missing scorer column {s}")
    return df


def assert_disjoint() -> None:
    """Calibration and test qids must never overlap. Checked whenever both question files exist."""
    ps = [path(s, "questions") for s in SPLITS]
    if not all(p.exists() for p in ps):
        return
    ids = [set(pd.read_csv(p, dtype=str)["qid"]) for p in ps]
    both = ids[0] & ids[1]
    if both:
        raise AssertionError(f"calibration and test share {len(both)} qids, e.g. {sorted(both)[:5]}")


def consensus(labels: pd.DataFrame) -> pd.DataFrame:
    """Per (qid, model): CORRECT or INCORRECT only if both judges agree, else AMBIGUOUS.

    Missing a judge counts as disagreement.
    """
    rows = []
    for (qid, model), g in labels.groupby(["qid", "model"], sort=True):
        v = dict(zip(g["judge"], g["verdict"]))
        a, b = v.get("judge_a"), v.get("judge_b")
        label = a if (a == b and a in ("CORRECT", "INCORRECT")) else "AMBIGUOUS"
        rows.append({"qid": qid, "model": model, "label": label, "judge_a": a, "judge_b": b})
    return pd.DataFrame(rows)


def load_split(split: str) -> dict:
    """Load a split and validate the contract. Returns questions, generations, labels, consensus."""
    assert_disjoint()
    q = read_questions(split)
    g = read_generations(split)
    lab = read_labels(split)
    qids = set(q["qid"])
    bad = set(g["qid"]) - qids
    if bad:
        raise ValueError(f"{split}: generations reference unknown qids {sorted(bad)[:5]}")
    bad = set(lab["qid"]) - qids
    if bad:
        raise ValueError(f"{split}: labels reference unknown qids {sorted(bad)[:5]}")
    dup = g.duplicated(["qid", "role"]).sum()
    if dup:
        raise ValueError(f"{split}: {dup} duplicate (qid, role) generation rows")
    return {"questions": q, "generations": g, "labels": lab, "consensus": consensus(lab),
            "hash": {k: file_hash(path(split, k)) for k in ("questions", "generations", "labels")}}
