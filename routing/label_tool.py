"""Blind human labeling in the terminal.

Shows only question, reference answer, and the cheap model's primary answer. Never a score, a judge
verdict, or a model name. Verdicts append immediately to data/<split>_human_labels.csv and the tool
resumes per annotator.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

HUMAN_COLS = ["qid", "annotator", "verdict", "note", "timestamp"]
KEYS = {"c": "CORRECT", "i": "INCORRECT", "a": "AMBIGUOUS"}
FORBIDDEN_CI = tuple(data.SCORERS) + ("judge_a", "judge_b", "score")  # matched case-insensitively
FORBIDDEN_CS = tuple(data.VERDICTS)  # the label-file tokens, matched as written
FORBIDDEN = FORBIDDEN_CI + FORBIDDEN_CS


def leaks(text: str) -> list[str]:
    """Forbidden tokens present in text: scorer and judge names (any case) and the upper-case verdicts."""
    low = text.lower()
    return [w for w in FORBIDDEN_CI if w in low] + [w for w in FORBIDDEN_CS if w in text]


def human_path(split: str) -> Path:
    return data.DATA / f"{split}_human_labels.csv"


def sample_path(split: str, seed: int) -> Path:
    return data.DATA / f"human_sample_{split}_{seed}.json"


def stratified_sample(split: str, n: int, seed: int) -> list[str]:
    """n qids with the split's consensus-label proportions (cheap model). Recorded to JSON once."""
    p = sample_path(split, seed)
    if p.exists():
        return json.loads(p.read_text())["qids"]
    d = data.load_split(split)
    cheap = data.models_config()["cheap"]
    c = d["consensus"]
    c = c[c["model"] == cheap].sort_values("qid")
    rng = np.random.RandomState(seed)
    counts = c["label"].value_counts()
    alloc = {lab: int(round(n * k / len(c))) for lab, k in counts.items()}
    while sum(alloc.values()) > n:
        alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < n:
        alloc[max(alloc, key=alloc.get)] += 1
    qids = []
    for lab, k in sorted(alloc.items()):
        pool = c[c["label"] == lab]["qid"].tolist()
        qids += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    qids = sorted(qids)
    p.write_text(json.dumps({"split": split, "seed": seed, "n": len(qids), "qids": qids}, indent=1))
    return qids


def render(question: str, reference: str, candidate: str) -> str:
    text = (f"\nQuestion: {question}\nReference answer: {reference}\nCandidate answer: {candidate}\n"
            "Your call [c / i / a], optional note after a space: ")
    bad = leaks(text)
    assert not bad, f"prompt leaks {bad}"
    return text


def _done(split: str, annotator: str) -> set:
    p = human_path(split)
    if not p.exists():
        return set()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return set(df[df["annotator"] == annotator]["qid"])


def _append(split: str, row: dict) -> None:
    p = human_path(split)
    new = not p.exists()
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HUMAN_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def run(split: str, n: int, seed: int, annotator: str, stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    qids = stratified_sample(split, n, seed)
    q = data.read_questions(split).set_index("qid")
    g = pd.read_csv(data.path(split, "generations"), dtype=str, keep_default_na=False)
    prim = g[g["role"] == "primary"].set_index("qid")["answer_text"]
    done = _done(split, annotator)
    todo = [x for x in qids if x not in done]
    labelled = 0
    for i, qid in enumerate(todo, 1):
        stdout.write(f"[{len(done) + i}/{len(qids)}]")
        stdout.write(render(q.loc[qid, "question"], q.loc[qid, "reference_answer"], prim[qid]))
        stdout.flush()
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        key, _, note = line.partition(" ")
        while key.lower() not in KEYS:
            stdout.write("Type c, i or a: ")
            stdout.flush()
            line = stdin.readline()
            if not line:
                return labelled
            key, _, note = line.strip().partition(" ")
        _append(split, {"qid": qid, "annotator": annotator, "verdict": KEYS[key.lower()], "note": note.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        labelled += 1
    stdout.write(f"\n{labelled} labelled this session, {len(done) + labelled}/{len(qids)} total for {annotator}\n")
    return labelled
