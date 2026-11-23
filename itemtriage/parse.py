"""Pure parsing of letter distributions and P(True) log-odds from top logprobs. Label-free."""

from __future__ import annotations

import math

from routing.scorers import p_true_logodds  # noqa: F401  (re-exported: same convention as routing)

from . import paths


def _lse(v: list) -> float:
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def letter_logprobs(top: list, n_options: int) -> tuple[list, int, bool]:
    """Per-letter log probabilities from a first-token top list of {token, logprob}.

    Tokens whose stripped, upper-cased text is a letter are pooled by logsumexp. A letter absent from the
    list takes the lowest returned logprob, an upper bound. Returns (logprobs renormalized over the
    letters, number of letters missing, whether the top-1 token is a letter).
    """
    letters = paths.LETTERS[:n_options]
    floor = min(t["logprob"] for t in top)
    raw = []
    missing = 0
    for L in letters:
        v = [t["logprob"] for t in top if t["token"].strip().upper() == L]
        if not v:
            missing += 1
        raw.append(_lse(v) if v else floor)
    z = _lse(raw)
    top1 = max(top, key=lambda t: t["logprob"])["token"].strip().upper()
    return [x - z for x in raw], missing, top1 in letters


def unpermute(logprobs: list, order: list) -> list:
    """Map logprobs over shown positions back to original option indices. order[i] = original index shown at i."""
    out = [0.0] * len(order)
    for shown_pos, orig in enumerate(order):
        out[orig] = logprobs[shown_pos]
    return out
