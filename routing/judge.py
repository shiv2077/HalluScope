"""Blind two-judge correctness labels. Judges see question, reference answer, candidate answer only.

Both judges are OpenAI models (the only provider available). Two judges from one provider share more
failure modes than two providers would; results/phase1_notes.md records this as a limitation.

This module never reads a generation's token list, token counts, or any routing signal.
Resumable per (qid, model, judge).
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

from . import data
from .cost import log_call, require_keys

RUBRIC = ("You are grading a short factual answer. You will see a question, a reference answer, and a "
          "candidate answer. Decide whether the candidate answer states the same fact as the reference. "
          "Ignore length, phrasing, and extra correct context. Mark INCORRECT if the candidate names a "
          "different entity, date, number, or gives the opposite answer. Mark AMBIGUOUS only if the "
          "reference itself is unclear or the candidate hedges without committing. Respond with exactly "
          "one line in the form VERDICT: <CORRECT|INCORRECT|AMBIGUOUS> and a second line REASON: <one sentence>.")
LABEL_COLS = ["qid", "model", "judge", "verdict", "reason"]
JUDGED_ROLES = ("primary", "strong")


def user_message(question: str, reference: str, candidate: str) -> str:
    return f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}"


def parse(text: str) -> tuple[str, str]:
    v = re.search(r"VERDICT:\s*(CORRECT|INCORRECT|AMBIGUOUS)", text, re.I)
    r = re.search(r"REASON:\s*(.+)", text, re.I | re.S)
    verdict = v.group(1).upper() if v else "AMBIGUOUS"
    reason = (r.group(1).strip().splitlines()[0] if r else f"unparsed: {text[:120]}").replace("\n", " ")
    return verdict, reason


def ask_openai(client, model: str, question: str, reference: str, candidate: str, purpose: str) -> str:
    with log_call(model, purpose) as rec:
        r = client.chat.completions.create(model=model, temperature=0, max_tokens=120, seed=42, messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_message(question, reference, candidate)}])
        rec["input_tokens"], rec["output_tokens"] = r.usage.prompt_tokens, r.usage.completion_tokens
    return r.choices[0].message.content or ""


def _existing(p: Path) -> set:
    if not p.exists():
        return set()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return set(zip(df["qid"], df["model"], df["judge"]))


def judge_split(split: str, judges: tuple = data.JUDGES) -> None:
    cfg = data.models_config()
    if cfg["judge_a"] == cfg["judge_b"]:
        raise SystemExit("judge_a and judge_b must be different models")
    require_keys("OPENAI_API_KEY")
    import openai

    client = openai.OpenAI()
    q = data.read_questions(split).set_index("qid")
    gens = pd.read_csv(data.path(split, "generations"), dtype=str, keep_default_na=False)
    gens = gens[gens["role"].isin(JUDGED_ROLES)][["qid", "model", "answer_text"]]
    out = data.path(split, "labels")
    done = _existing(out)
    todo = [(g.qid, g.model, g.answer_text, j) for g in gens.itertuples() for j in judges
            if (g.qid, g.model, j) not in done]
    print(f"{split}: {len(done)} labels exist, {len(todo)} to judge", file=sys.stderr)
    for i, (qid, model, candidate, j) in enumerate(todo, 1):
        question, reference = q.loc[qid, "question"], q.loc[qid, "reference_answer"]
        text = ask_openai(client, cfg[j], question, reference, candidate, f"{split}:{j}")
        verdict, reason = parse(text)
        new = not out.exists()
        with open(out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LABEL_COLS)
            if new:
                w.writeheader()
            w.writerow({"qid": qid, "model": model, "judge": j, "verdict": verdict, "reason": reason})
        if i % 50 == 0:
            print(f"{split}: {i}/{len(todo)}", file=sys.stderr)
