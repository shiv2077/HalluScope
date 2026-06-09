"""Exact-match grading against TriviaQA-style alias lists (CLAUDE.md).

Normalization mirrors the SQuAD/TriviaQA convention: lowercase, drop
punctuation, drop articles (a/an/the), collapse whitespace. A generation is
graded *correct* if any normalized gold alias appears as a contiguous token
subsequence of the normalized model answer (handles verbose answers like
"The capital is Paris."), or exactly equals it.

Every run logs 50 random grading decisions for human audit.
"""

from __future__ import annotations

import csv
import dataclasses
import random
import re
import string
from pathlib import Path

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    """SQuAD/TriviaQA normalization: lower, strip punct + articles, collapse ws."""
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize(s).split()


def _contains_subseq(hay: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    n, m = len(hay), len(needle)
    for i in range(n - m + 1):
        if hay[i : i + m] == needle:
            return True
    return False


def alias_em(prediction: str, aliases: list[str]) -> tuple[bool, str | None]:
    """Return (is_correct, matched_alias).

    Correct iff a normalized alias == normalized prediction, or appears as a
    contiguous token subsequence within it.
    """
    pred_norm = normalize(prediction)
    pred_toks = pred_norm.split()
    for alias in aliases:
        a_norm = normalize(alias)
        if not a_norm:
            continue
        if a_norm == pred_norm:
            return True, alias
        if _contains_subseq(pred_toks, a_norm.split()):
            return True, alias
    return False, None


@dataclasses.dataclass
class GraderAudit:
    """Collect grading decisions; dump a 50-row random sample for human review."""

    rows: list[dict] = dataclasses.field(default_factory=list)

    def record(self, qid: str, prediction: str, aliases: list[str], correct: bool, matched: str | None):
        self.rows.append(
            {
                "qid": qid,
                "prediction": prediction[:300],
                "aliases": " | ".join(aliases[:8]),
                "correct": int(correct),
                "matched_alias": matched or "",
            }
        )

    def dump_sample(self, path: str | Path, n: int = 50, seed: int = 42) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        sample = self.rows if len(self.rows) <= n else rng.sample(self.rows, n)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qid", "prediction", "aliases", "correct", "matched_alias"])
            w.writeheader()
            w.writerows(sample)
        return path


def grade(prediction: str, aliases: list[str], qid: str = "", audit: GraderAudit | None = None) -> bool:
    correct, matched = alias_em(prediction, aliases)
    if audit is not None:
        audit.record(qid, prediction, aliases, correct, matched)
    return correct
