"""
HalluScope – HRI (Hallucination Risk Index) Engine

Combines token-level uncertainty metrics into a single scalar risk score via
a weighted sigmoid. Adapter-agnostic: it operates on the StandardizedResult
produced by any LLM adapter.

No magic numbers
----------------
All weights, the sigmoid scale/bias, the normalization constants, and the
risk-band thresholds are loaded from ``config/*.json`` through
``halluscope_core.config_store`` (re-exported to the FastAPI app as
``backend.app.config_store`` for backward compatibility). This file contains
NO numeric risk constants:

  * combination weights + sigmoid params  -> config/weights.json
  * rcd_scale (+ other norms)              -> config/norm.json
  * risk_low / risk_high band cutoffs      -> config/thresholds.json
    (risk_high is shared with intervention_service — one source of truth)

HRI = sigmoid( scale * sum_i(w_i * feature_i) + bias )

Semantic-energy direction
-------------------------
The local-mode ``semantic_energy`` feature is the bounded flatness signal in
[0, 1] produced by ``internal_metrics.semantic_energy`` (flatter logits =>
higher value => higher risk). It is used directly — there is no abs() and no
divisor (the old abs(E)/35 inverted the sign; see internal_metrics docstring).
"""

import math
from typing import List, Dict, Any, Optional

from halluscope_core import config_store
from halluscope_core.metrics.logprob_metrics import (
    mean_logprob,
    logprob_variance,
    logprob_std,
)
from halluscope_core.metrics.d import confidence_decay


def _sigmoid(x: float) -> float:
    """Standard sigmoid activation, clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _weighted_raw(features: Dict[str, float], wmap: Dict[str, Any]) -> float:
    """Dot-product of the named feature vector with the named weight map.

    Only keys present in ``wmap`` AND numeric contribute; meta keys such as
    ``sigmoid`` / ``_doc`` are ignored.
    """
    total = 0.0
    for key, w in wmap.items():
        if key in features and isinstance(w, (int, float)):
            total += float(w) * float(features[key])
    return total


# Canonical feature names per mode (shared with config/weights.json and the
# benchmark feature columns consumed by eval/fit_weights.py).
FEATURE_NAMES_API = ["logprob_variance", "confidence_decay", "abs_mean_logprob"]
FEATURE_NAMES_LOCAL = FEATURE_NAMES_API + ["semantic_energy", "rcd_inv"]


def hri_from_features(
    features: Dict[str, float],
    mode: str = "api",
    weights: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute HRI from a named feature vector and a weight map.

    Lets fit_weights / ablation / cross-model transfer recompute HRI from
    logged feature columns under arbitrary weights without re-running the LLM.
    ``weights`` defaults to config/weights.json; ``mode`` selects the
    ``"api"`` / ``"local"`` weight block.
    """
    w = weights or config_store.weights()
    wmap = w[mode]
    # A fitted mode block may carry its own sigmoid (so fitting one mode does
    # not disturb the other); otherwise fall back to the shared sigmoid.
    sig = wmap.get("sigmoid", w["sigmoid"]) if isinstance(wmap, dict) else w["sigmoid"]
    scale = float(sig["scale"])
    bias = float(sig["bias"])
    raw = _weighted_raw(features, wmap)
    return _sigmoid(scale * raw + bias)


def classify_risk(hri: float) -> str:
    """Map an HRI score to a LOW / MEDIUM / HIGH band.

    Reads the SINGLE source of truth in config/thresholds.json. The same
    ``risk_high`` is used by intervention_service.should_intervene, so a
    response classified HIGH is exactly the set of responses that trigger
    intervention.
    """
    low = config_store.risk_low()
    high = config_store.risk_high()
    if hri < low:
        return "LOW"
    if hri < high:
        return "MEDIUM"
    return "HIGH"


def _token_entropy(top_lps: List[float]) -> float:
    """Shannon entropy from top-K log-probs for a single token.

    NOTE: this is a top-K (K=5) truncated entropy, computed over the provided
    alternates only — it is NOT the full-vocabulary entropy. Both the OpenAI
    and local adapters surface only top-5 alternates here. (A full-vocab
    entropy is available locally and is exposed separately as the semantic
    energy flatness signal.)
    """
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

    API-mode (default) uses logprob-derived features only. When
    ``internal_signals`` carries non-None ``mean_semantic_energy`` and
    ``mean_rcd``, the local-mode feature set (which adds the flatness energy
    and inverted representation-collapse terms) is used instead, and
    ``hri_mode`` is reported as ``"local_internal"`` vs ``"api"``.

    Returns: hri, hri_mode, risk_level, metrics, per_token_hri,
    risk_contributions, token_entropies.
    """
    w = config_store.weights()
    norm = config_store.norm()
    scale = float(w["sigmoid"]["scale"])
    bias = float(w["sigmoid"]["bias"])

    n = len(logprobs)

    # ── Basic metrics ────────────────────────────────────────────────
    m_mean = mean_logprob(logprobs)
    m_var = logprob_variance(logprobs)
    m_std = logprob_std(logprobs)
    m_decay = confidence_decay(logprobs)

    # ── Entropy metrics (top-K truncated) ────────────────────────────
    token_entropies: List[float] = []
    if top_logprobs:
        for top_lps in top_logprobs:
            token_entropies.append(_token_entropy(top_lps))
    avg_entropy = (
        sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
    )
    max_entropy = math.log(5) if top_logprobs else 1.0  # log(K) for K=5
    norm_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0

    # ── NLL & Perplexity ─────────────────────────────────────────────
    nll = -m_mean
    perplexity = round(min(math.exp(min(nll, 20)), 1e9), 4)

    # ── Feature vector (shared names with config/weights.json) ───────
    features: Dict[str, float] = {
        "logprob_variance": m_var,
        "confidence_decay": m_decay,
        "abs_mean_logprob": abs(m_mean),
    }

    hri_mode = "api"
    wmap = w["api"]

    if internal_signals is not None:
        mean_energy = internal_signals.get("mean_semantic_energy")
        mean_rcd = internal_signals.get("mean_rcd")
        if mean_energy is not None and mean_rcd is not None:
            # semantic_energy is already a flatness value in [0, 1]
            # (internal_metrics.semantic_energy): use it directly, clamped.
            energy_feat = min(1.0, max(0.0, float(mean_energy)))
            # Low RCD => representation collapse => risk. Invert so HIGH
            # inverted-RCD = HIGH risk. rcd_scale maps [0, ~0.5] -> [0, 1].
            rcd_scale = float(norm["rcd_scale"])
            rcd_inv = 1.0 - min(float(mean_rcd) * rcd_scale, 1.0)
            features["semantic_energy"] = energy_feat
            features["rcd_inv"] = rcd_inv
            wmap = w["local"]
            hri_mode = "local_internal"

    # Honor a per-mode sigmoid if a fitted weight block carries one.
    mode_sig = wmap.get("sigmoid") if isinstance(wmap, dict) else None
    m_scale = float(mode_sig["scale"]) if mode_sig else scale
    m_bias = float(mode_sig["bias"]) if mode_sig else bias
    raw = _weighted_raw(features, wmap)
    hri = _sigmoid(m_scale * raw + m_bias)

    risk_level = classify_risk(hri)

    # ── Per-token HRI trajectory (logprob-only / api weights) ────────
    api_w = w["api"]
    per_token_hri: List[float] = []
    for i, lp in enumerate(logprobs):
        window = logprobs[: i + 1]
        w_feats = {
            "logprob_variance": logprob_variance(window) if len(window) >= 2 else 0.0,
            "abs_mean_logprob": abs(mean_logprob(window)),
            "confidence_decay": confidence_decay(window) if len(window) >= 2 else 0.0,
        }
        t_raw = _weighted_raw(w_feats, api_w)
        per_token_hri.append(round(_sigmoid(scale * t_raw + bias), 4))

    # ── Per-token risk contribution (for heatmap) ────────────────────
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
