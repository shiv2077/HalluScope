"""
HalluScope v2 – HRI (Hallucination Risk Index) Engine

Combines multiple token-level uncertainty metrics into a single
scalar risk score using a weighted sigmoid formula.

The engine is **adapter-agnostic**: it operates purely on the
StandardizedResult produced by any LLM adapter.

v2 changes:
  - New thresholds: LOW < 0.35, MEDIUM 0.35–0.45, HIGH >= 0.46
  - Extended metrics: entropy, normalized entropy, NLL, perplexity
  - Per-token HRI trajectory for charting
  - Per-token risk contribution for heatmap
"""

import math
from typing import List, Dict, Any

from backend.app.metrics.logprob_metrics import (
    mean_logprob,
    logprob_variance,
    logprob_std,
)
from backend.app.metrics.confidence_metrics import confidence_decay


# ── Risk-level thresholds (v2) ───────────────────────────────────────
RISK_LOW_THRESHOLD = 0.35
RISK_HIGH_THRESHOLD = 0.46


def _sigmoid(x: float) -> float:
    """Standard sigmoid activation, clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _token_entropy(top_lps: List[float]) -> float:
    """Shannon entropy from top-K log-probs for a single token."""
    if not top_lps:
        return 0.0
    probs = [math.exp(lp) for lp in top_lps]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def compute_hri(
    logprobs: List[float],
    top_logprobs: List[List[float]] | None = None,
) -> Dict[str, Any]:
    """Compute the Hallucination Risk Index and full metrics suite.

    Formula:
        HRI = sigmoid(
            0.4 * logprob_variance
          + 0.3 * confidence_decay
          + 0.3 * |mean_logprob|
        )

    Returns a dict with hri, risk_level, metrics, per_token_hri,
    and per_token_risk_contribution.
    """

    n = len(logprobs)

    # ── Basic metrics ────────────────────────────────────────────────
    m_mean = mean_logprob(logprobs)
    m_var = logprob_variance(logprobs)
    m_std = logprob_std(logprobs)
    m_decay = confidence_decay(logprobs)

    # ── Entropy metrics ──────────────────────────────────────────────
    token_entropies: List[float] = []
    if top_logprobs:
        for top_lps in top_logprobs:
            token_entropies.append(_token_entropy(top_lps))
    avg_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
    max_entropy = math.log(5) if top_logprobs else 1.0  # log(K) for K=5
    norm_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0

    # ── NLL & Perplexity ─────────────────────────────────────────────
    nll = -m_mean  # negative log likelihood = -mean(logprob)
    perplexity = math.exp(nll) if nll < 20 else float("inf")

    # ── Composite HRI score ──────────────────────────────────────────
    raw = 0.4 * m_var + 0.3 * m_decay + 0.3 * abs(m_mean)
    # Shifted sigmoid: raw=0 → ~0.05 (LOW), real uncertainty → HIGH
    hri = _sigmoid(10.0 * raw - 3.0)

    # ── Risk classification ──────────────────────────────────────────
    if hri < RISK_LOW_THRESHOLD:
        risk_level = "LOW"
    elif hri < RISK_HIGH_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # ── Per-token HRI trajectory ─────────────────────────────────────
    per_token_hri: List[float] = []
    for i, lp in enumerate(logprobs):
        window = logprobs[: i + 1]
        w_var = logprob_variance(window) if len(window) >= 2 else 0.0
        w_mean = mean_logprob(window)
        w_decay = confidence_decay(window) if len(window) >= 2 else 0.0
        t_raw = 0.4 * w_var + 0.3 * w_decay + 0.3 * abs(w_mean)
        per_token_hri.append(round(_sigmoid(10.0 * t_raw - 3.0), 4))

    # ── Per-token risk contribution ──────────────────────────────────
    risk_contributions: List[float] = []
    for i, lp in enumerate(logprobs):
        ent = token_entropies[i] if i < len(token_entropies) else 0.0
        contrib = 0.5 * abs(lp) + 0.5 * ent
        risk_contributions.append(round(contrib, 4))

    return {
        "hri": round(hri, 4),
        "risk_level": risk_level,
        "metrics": {
            "mean_logprob": round(m_mean, 4),
            "logprob_std": round(m_std, 4),
            "logprob_variance": round(m_var, 4),
            "entropy": round(avg_entropy, 4),
            "normalized_entropy": round(norm_entropy, 4),
            "neg_log_likelihood": round(nll, 4),
            "perplexity": round(perplexity, 4),
            "confidence_decay": round(m_decay, 4),
            "token_count": n,
        },
        "per_token_hri": per_token_hri,
        "risk_contributions": risk_contributions,
        "token_entropies": [round(e, 4) for e in token_entropies],
    }
