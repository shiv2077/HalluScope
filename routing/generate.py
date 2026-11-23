"""Build question files and generate every role for both splits. Resumable per (qid, role).

Intervention does not exist on this path: raw chat completion calls, real answer text always stored.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

from . import data
from .cost import log_call, require_keys

SYSTEM = "Answer the question concisely."
PTRUE_TEMPLATE = ("Question: {q}\nProposed answer: {a}\n"
                  "Is the proposed answer correct? Answer with exactly one word, Yes or No.")
WARNING_BANNER = "⚠️ **High hallucination risk detected."
GEN_COLS = ["qid", "model", "role", "answer_text", "token_logprobs", "input_tokens", "output_tokens",
            "temperature", "seed"]


# ---------------------------------------------------------------- question files
def build_calibration_questions() -> pd.DataFrame:
    """82 real-answer rows of data/routing/results_api.csv plus the 80 gold rows. Template rows dropped."""
    api = pd.read_csv(data.ROOT / "data" / "routing" / "results_api.csv", dtype=str, keep_default_na=False)
    api = api[~api["model_answer"].str.startswith(WARNING_BANNER)]
    gold = pd.read_csv(data.ROOT / "data" / "routing" / "gold_results_api.csv", dtype=str, keep_default_na=False)
    q = pd.concat([api[["id", "question", "reference_answer"]], gold[["id", "question", "reference_answer"]]])
    q = q.rename(columns={"id": "qid"}).reset_index(drop=True)
    assert len(q) == 162, len(q)
    assert q["qid"].is_unique
    return q


def build_test_questions(n: int = 300, seed: int = 42) -> pd.DataFrame:
    items = json.loads((data.ROOT / "data" / "routing" / "halueval_qa_test.json").read_text())
    random.Random(seed).shuffle(items)
    q = pd.DataFrame(items[:n])[["id", "question", "reference_answer"]].rename(columns={"id": "qid"})
    return q


def build_questions() -> None:
    data.DATA.mkdir(parents=True, exist_ok=True)
    cal = build_calibration_questions()
    test = build_test_questions()
    both = set(cal["qid"]) & set(test["qid"])
    assert not both, both
    assert not (set(cal["question"]) & set(test["question"])), "question text overlap"
    cal.to_csv(data.path("calibration", "questions"), index=False)
    test.to_csv(data.path("test", "questions"), index=False)
    data.assert_disjoint()
    print(f"wrote {len(cal)} calibration and {len(test)} test questions")


# ---------------------------------------------------------------- API calls
def _client():
    import openai

    return openai.OpenAI()


def chat(client, model: str, messages: list, purpose: str, temperature: float, seed: int,
         max_tokens: int, logprobs: bool):
    with log_call(model, purpose) as rec:
        r = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                           seed=seed, max_tokens=max_tokens, logprobs=logprobs,
                                           top_logprobs=5 if logprobs else None)
        rec["input_tokens"] = r.usage.prompt_tokens
        rec["output_tokens"] = r.usage.completion_tokens
    ch = r.choices[0]
    text = ch.message.content or ""
    if logprobs and ch.logprobs and ch.logprobs.content:
        toks = [round(t.logprob, 6) for t in ch.logprobs.content]
        first_top = [{"token": a.token, "logprob": round(a.logprob, 6)} for a in ch.logprobs.content[0].top_logprobs]
    else:
        toks, first_top = [], []
    return text, toks, first_top, r.usage.prompt_tokens, r.usage.completion_tokens


def _existing(p: Path) -> set:
    if not p.exists():
        return set()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return set(zip(df["qid"], df["role"]))


def _append(p: Path, row: dict) -> None:
    new = not p.exists()
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GEN_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def generate_split(split: str, roles: tuple = data.ROLES) -> None:
    require_keys("OPENAI_API_KEY")
    cfg = data.models_config()
    client = _client()
    q = data.read_questions(split)
    out = data.path(split, "generations")
    done = _existing(out)
    todo = [(r.qid, r.question, r.reference_answer, role) for r in q.itertuples() for role in roles
            if (r.qid, role) not in done]
    print(f"{split}: {len(done)} rows exist, {len(todo)} to generate", file=sys.stderr)
    primary_cache: dict = {}
    for i, (qid, question, _ref, role) in enumerate(todo, 1):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        if role == "primary":
            model, temp, seed, lp = cfg["cheap"], cfg["primary_temperature"], cfg["seed"], True
        elif role.startswith("resample"):
            model, temp, seed, lp = cfg["cheap"], cfg["resample_temperature"], cfg["seed"] + int(role[-1]), False
        elif role == "strong":
            model, temp, seed, lp = cfg["strong"], cfg["primary_temperature"], cfg["seed"], False
        elif role == "ptrue":
            model, temp, seed, lp = cfg["cheap"], 0, cfg["seed"], True
            prim = primary_cache.get(qid) or _primary_text(out, qid)
            if prim is None:
                raise RuntimeError(f"ptrue for {qid} needs the primary row first; roles are ordered")
            msgs = [{"role": "user", "content": PTRUE_TEMPLATE.format(q=question, a=prim)}]
        else:
            raise ValueError(role)
        text, toks, first_top, ti, to = chat(client, model, msgs, f"{split}:{role}", temp, seed,
                                             cfg["max_tokens"] if role != "ptrue" else 3, lp)
        if role == "primary":
            primary_cache[qid] = text
        lp_json = json.dumps(first_top if role == "ptrue" else toks)
        _append(out, {"qid": qid, "model": model, "role": role, "answer_text": text, "token_logprobs": lp_json,
                      "input_tokens": ti, "output_tokens": to, "temperature": temp, "seed": seed})
        if i % 25 == 0:
            print(f"{split}: {i}/{len(todo)}", file=sys.stderr)


def _primary_text(p: Path, qid: str):
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    m = df[(df["qid"] == qid) & (df["role"] == "primary")]
    return None if m.empty else m["answer_text"].iloc[0]
