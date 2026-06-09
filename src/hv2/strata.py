"""Strata assignment: KH / KC / UH / UC (CLAUDE.md).

Band a question by its knowledge probe score, then label each *generation* by
whether it was correct:

  K = knows      (n_correct_probes >= 6/8)        U = doesn't know (<= 2/8)
  C = correct generation                          H = hallucination (wrong generation)

  KH = knows but hallucinates  <- the scientifically scarce, central stratum
  KC = knows and correct
  UH = doesn't know and wrong
  UC = doesn't know and (accidentally) correct
  MID = middle band (2/8 < score < 6/8) -> excluded from the core analysis

Each perturbed generation is one data point; the strata table is broken down by
perturbation type so we can see, e.g., how many KH cases each of P1/P2/P3 yields.
"""

from __future__ import annotations

from collections import Counter

from .probes import know_band

CORE_STRATA = ("KH", "KC", "UH", "UC")


def assign_stratum(n_correct_probes: int, generation_correct: bool, n_probes: int = 8) -> str:
    band = know_band(n_correct_probes, n_probes)
    if band == "mid":
        return "MID"
    k = "K" if band == "high" else "U"
    h = "C" if generation_correct else "H"
    return k + h


def strata_table(records: list[dict]) -> dict:
    """records: dicts with keys {n_correct_probes, correct, perturb_type}.

    Returns overall counts and a per-perturbation breakdown.
    """
    overall: Counter = Counter()
    by_type: dict[str, Counter] = {}
    for r in records:
        s = assign_stratum(r["n_correct_probes"], r["correct"])
        overall[s] += 1
        pt = r.get("perturb_type", "neutral")
        by_type.setdefault(pt, Counter())[s] += 1
    return {
        "overall": {s: overall.get(s, 0) for s in (*CORE_STRATA, "MID")},
        "by_perturbation": {pt: {s: c.get(s, 0) for s in (*CORE_STRATA, "MID")} for pt, c in by_type.items()},
        "n": len(records),
    }
