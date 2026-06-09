"""Stage 1 — pre-generation prompt-risk signals (s1..s4).

These are computed from the PROMPT alone, before/around generation, and
estimate how likely the question is to induce a hallucination regardless of
what the model answers. Each signal is bounded to [0, 1] (higher = riskier).
They are aggregated with a configurable sigmoid into a single ``prompt_risk``
scalar.

  s1 entity_rarity     — fraction of rare named entities in the prompt.
  s2 time_risk         — post-cutoff years / relative temporal markers.
  s3 prompt_perplexity — perplexity of the PROMPT tokens (local mode). In API
                         mode true prompt perplexity is unavailable; the caller
                         passes None and this signal is a neutral 0.0 (the
                         honest API-mode fallback is documented at the call
                         site — generation perplexity is NOT prompt perplexity).
  s4 quoted_title      — presence of quoted title-like phrases.

Weights/bias: config/weights.json -> prompt_risk. Params (cutoff year,
relative markers): config/prompt_risk.json. No magic numbers here.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from backend.app import config_store
from backend.app.metrics.nlp_backend import rare_entity_fraction

_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
# Quoted spans: straight or curly double/single quotes.
_QUOTED = re.compile(r"[\"“]([^\"”]{2,})[\"”]|[\'‘]([^\'’]{2,})[\'’]")


def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ── s1 ───────────────────────────────────────────────────────────────
def entity_rarity_signal(prompt: str) -> float:
    """Fraction of rare named-entity tokens in the prompt (∈ [0, 1])."""
    return rare_entity_fraction(prompt)


# ── s2 ───────────────────────────────────────────────────────────────
def time_risk(prompt: str, knowledge_cutoff_year: Optional[int] = None) -> float:
    """Temporal risk of the prompt (∈ [0, 1]).

    A year strictly after the model's knowledge cutoff (=> unknowable) is the
    strongest signal; relative markers ("today", "latest", ...) are moderate;
    any explicit year is a mild signal.
    """
    cfg = config_store.prompt_risk()
    cutoff = (
        knowledge_cutoff_year
        if knowledge_cutoff_year is not None
        else int(cfg["knowledge_cutoff_year"])
    )
    markers: List[str] = [m.lower() for m in cfg["relative_markers"]]

    lower = prompt.lower()
    years = [int(y) for y in _YEAR.findall(prompt)]
    post_cutoff = any(y > cutoff for y in years)
    has_year = len(years) > 0
    has_relative = any(m in lower for m in markers)

    score = 0.0
    if post_cutoff:
        score += 0.6
    if has_relative:
        score += 0.3
    if has_year:
        score += 0.2
    return min(1.0, score)


# ── s3 ───────────────────────────────────────────────────────────────
def prompt_perplexity(token_logprobs: List[float]) -> float:
    """Perplexity = exp(-mean logprob) of the provided token logprobs."""
    if not token_logprobs:
        return 0.0
    mean_lp = sum(token_logprobs) / len(token_logprobs)
    return math.exp(min(-mean_lp, 20.0))


def prompt_perplexity_signal(
    prompt_token_logprobs: Optional[List[float]],
    divisor: Optional[float] = None,
) -> float:
    """Normalised prompt-perplexity signal (∈ [0, 1]).

    ``None`` / empty (API mode, where prompt-token logprobs are unavailable)
    => neutral 0.0. Otherwise perplexity / divisor, clipped to 1.0.
    """
    if not prompt_token_logprobs:
        return 0.0
    if divisor is None:
        divisor = float(config_store.norm()["prompt_perplexity_divisor"])
    ppl = prompt_perplexity(prompt_token_logprobs)
    return min(1.0, ppl / divisor)


# ── s4 ───────────────────────────────────────────────────────────────
def quoted_title_risk(prompt: str) -> float:
    """Risk from quoted title-like phrases (∈ [0, 1]).

    Counts quoted spans (straight/curly quotes) and weights those that look
    like titles (contain a Title-Case word) more strongly.
    """
    score = 0.0
    for m in _QUOTED.finditer(prompt):
        span = m.group(1) or m.group(2) or ""
        looks_titley = any(w[:1].isupper() for w in span.split())
        score += 0.5 if looks_titley else 0.25
    return min(1.0, score)


# ── aggregate ────────────────────────────────────────────────────────
def compute_prompt_risk(
    prompt: str,
    prompt_token_logprobs: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute s1..s4 and the aggregate prompt_risk scalar.

    prompt_risk = sigmoid(scale * sum_i(w_i * s_i) + bias), with weights from
    config/weights.json -> prompt_risk.
    """
    w = config_store.weights()["prompt_risk"]
    scale = float(w["sigmoid"]["scale"])
    bias = float(w["sigmoid"]["bias"])

    s1 = entity_rarity_signal(prompt)
    s2 = time_risk(prompt)
    s3 = prompt_perplexity_signal(prompt_token_logprobs)
    s4 = quoted_title_risk(prompt)

    raw = (
        float(w["entity_rarity"]) * s1
        + float(w["time_risk"]) * s2
        + float(w["prompt_perplexity"]) * s3
        + float(w["quoted_title"]) * s4
    )
    prompt_risk_score = _sigmoid(scale * raw + bias)

    return {
        "s1_entity_rarity": round(s1, 4),
        "s2_time_risk": round(s2, 4),
        "s3_prompt_perplexity": round(s3, 4),
        "s4_quoted_title": round(s4, 4),
        "prompt_risk": round(prompt_risk_score, 4),
        "prompt_perplexity_available": bool(prompt_token_logprobs),
    }
