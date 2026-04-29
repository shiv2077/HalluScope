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
from typing import List, Dict, Any, Optional

from backend.app.metrics.logprob_metrics import (
    mean_logprob,
    logprob_variance,
    logprob_std,
)
from backend.app.metrics.confidence_metrics import confidence_decay


# ── Risk thresholds ──────────────────────────────────────────────────
# Calibrated on HalluEval QA (n=100, free-form-answer methodology).
# API-mode AUROC: 0.785
# Best-F1 threshold (API mode): 0.086 (precision=0.74, recall=0.89, F1=0.81)
# Selected operating point (precision-floor):
#   HIGH = 0.30  (precision=0.879, recall=0.468, flag-rate=33%)
#   LOW  = max(0.10, HIGH - 0.15)
# Rationale: best-F1 fires on too high a fraction of normal traffic
# (HalluEval is 62% adversarial-by-design). The selected operating
# point maintains precision >= 0.85 while keeping the flag rate
# below 50% on the calibration set.
# Calibration date: 2026-04-29
# See eval/results_api.csv, eval/roc_api.png, eval/REPORT.md.
# NOTE: thresholds and operating point chosen on the same set
# evaluated; held-out validation is scoped for next iteration.
# NOTE (fallback heuristic): no operating point in the calibration sweep
# (0.05..0.95 step 0.01) met precision >= 0.85 AND recall >= 0.50 AND
# flag-rate <= 0.50 simultaneously. The Pareto frontier on this n=100
# sample tops out near P=0.84/R=0.50; HIGH=0.30 is the closest defensible
# precision-floor point (P=0.88, R=0.47, flag-rate=33%) and is the
# fallback prescribed by eval/pick_thresholds.py.
RISK_LOW_THRESHOLD  = 0.15
RISK_HIGH_THRESHOLD = 0.30

# ── Local-mode normalisation constants ───────────────────────────────
# Heuristic pre-calibration values; full calibration against HalluEval is
# Phase 3 work. The denominators are chosen to map typical observed ranges
# of the underlying signals into roughly [0, 1] so they sit on the same
# scale as the existing logprob-derived terms.
SEMANTIC_ENERGY_DIVISOR = 35.0   # observed TinyLlama range: |E| ~ 25–35
                                 # (logsumexp grows with vocab; 32k vocab
                                 # gives larger E than the original 5–15
                                 # estimate — verified end-to-end on the
                                 # photosynthesis prompt). Phase 3 calibrates.
RCD_DOUBLER = 2.0                # typical RCD is ~0–0.5; doubling spans ~0–1


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
    internal_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the Hallucination Risk Index and full metrics suite.

    Default (API-mode) formula:
        HRI = sigmoid(10 * (
              0.4 * logprob_variance
            + 0.3 * confidence_decay
            + 0.3 * |mean_logprob|
        ) - 3)

    When internal_signals is provided AND its mean_semantic_energy and
    mean_rcd are both non-None, a 5-term local-mode formula is used
    instead. The result dict gains an "hri_mode" field of either
    "api" or "local_internal" so the frontend can label which formula
    was applied.

    Returns a dict with hri, hri_mode, risk_level, metrics,
    per_token_hri, per_token_risk_contribution, and token_entropies.
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
    # Cap perplexity to a finite, JSON-safe value: float('inf') serializes as
    # the literal "Infinity", which JSON.parse in the browser rejects.
    perplexity = round(min(math.exp(min(nll, 20)), 1e9), 4)

    # ── Composite HRI score ──────────────────────────────────────────
    # Default: API-mode formula (logprob-derived signals only).
    raw = 0.4 * m_var + 0.3 * m_decay + 0.3 * abs(m_mean)
    hri_mode = "api"

    # Local-mode override: when internal signals are present, swap in a
    # 5-term formula that incorporates Semantic Energy and Representation
    # Collapse Detection. Weights are heuristic pre-calibration values
    # (Phase 3 calibrates against HalluEval).
    if internal_signals is not None:
        mean_energy = internal_signals.get("mean_semantic_energy")
        mean_rcd = internal_signals.get("mean_rcd")
        if mean_energy is not None and mean_rcd is not None:
            # Normalise both signals into ~[0, 1]:
            #   Semantic energy is reported as -logsumexp(logits); larger
            #   |E| → flatter logits → more uncertainty. Use absolute value
            #   over the divisor and clip.
            energy_norm = min(abs(mean_energy) / SEMANTIC_ENERGY_DIVISOR, 1.0)
            #   RCD ∈ [0, 2] but typically [0, 0.5]; LOW RCD = collapse =
            #   bad, so invert so HIGH inverted-RCD = HIGH risk.
            rcd_inv = 1.0 - min(mean_rcd * RCD_DOUBLER, 1.0)

            raw = (
                0.30 * m_var
                + 0.20 * m_decay
                + 0.20 * abs(m_mean)
                + 0.15 * energy_norm
                + 0.15 * rcd_inv
            )
            hri_mode = "local_internal"

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
        "hri_mode": hri_mode,
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
