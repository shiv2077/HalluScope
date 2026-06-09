"""Knowledge probes: 8 neutral probes per question (CLAUDE.md).

4 paraphrase templates decoded GREEDILY + 4 temperature-0.7 SAMPLES of the base
template = 8 probes. ``know_score = #correct / 8``. A question is "known" (high
band) at >=6/8 and "unknown" (low band) at <=2/8 (== wrong on >=6/8 probes).

These templates are deliberately *neutral* — no leading context, no pressure.
Perturbations (perturb.py) are the opposite: they push the model to fail.
"""

from __future__ import annotations

import dataclasses

# 4 neutral paraphrase templates. {q} is the verbatim question.
PARAPHRASE_TEMPLATES = [
    "{q}",
    "Answer this question: {q}",
    "Question: {q}\nAnswer:",
    "Please answer the following trivia question concisely.\n{q}",
]

N_TEMP_SAMPLES = 4
TEMP_SEEDS = [101, 102, 103, 104]  # fixed seeds for the 4 temperature samples
PROBE_TEMPERATURE = 0.7
KNOW_HIGH = 6  # >=6/8 correct -> "knows"
KNOW_LOW = 2  # <=2/8 correct -> "doesn't know"


@dataclasses.dataclass(frozen=True)
class Probe:
    probe_id: str  # e.g. "para0", "temp2"
    text: str  # user-message content
    greedy: bool
    temperature: float | None
    seed: int | None


def build_probes(question: str) -> list[Probe]:
    """Return the 8 probes for a question (4 greedy paraphrases + 4 temp samples)."""
    probes: list[Probe] = []
    for i, tpl in enumerate(PARAPHRASE_TEMPLATES):
        probes.append(Probe(f"para{i}", tpl.format(q=question), greedy=True, temperature=None, seed=None))
    base = PARAPHRASE_TEMPLATES[0].format(q=question)
    for j in range(N_TEMP_SAMPLES):
        probes.append(Probe(f"temp{j}", base, greedy=False, temperature=PROBE_TEMPERATURE, seed=TEMP_SEEDS[j]))
    assert len(probes) == 8
    return probes


def know_score(correct_flags: list[bool]) -> float:
    """Fraction correct over the probes (expects 8)."""
    return sum(1 for c in correct_flags if c) / max(len(correct_flags), 1)


def know_band(n_correct: int, n: int = 8) -> str:
    """'high' (>=6/8), 'low' (<=2/8), else 'mid'."""
    if n_correct >= KNOW_HIGH:
        return "high"
    if n_correct <= KNOW_LOW:
        return "low"
    return "mid"
