"""Internal-signal metrics for local-mode HRI.

These metrics require access to model internals (full pre-softmax logits
and last-layer hidden states) and therefore cannot be computed for
API-based providers like OpenAI.

Implemented:
  - Semantic Energy:           E_t = -log sum(exp(z_t))
  - Representation Collapse:   RCD_t = 1 - cos(h_t, h_{t-1})
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


def semantic_energy(logits_per_token: List[np.ndarray]) -> List[float]:
    """Per-token semantic energy: E_t = -log sum_i exp(z_t,i).

    Higher energy = flatter logit distribution = more uncertainty.
    Uses the log-sum-exp trick for numerical stability so that very
    large logits do not overflow exp.
    """
    energies: List[float] = []
    for z in logits_per_token:
        z_max = float(np.max(z))
        # log sum exp = z_max + log sum exp(z - z_max)
        lse = z_max + math.log(float(np.sum(np.exp(z - z_max))))
        energies.append(float(-lse))
    return energies


def representation_collapse(hidden_states: List[np.ndarray]) -> List[float]:
    """Per-token RCD: 1 - cos(h_t, h_{t-1}).

    Returns a list of length len(hidden_states) - 1 (no value for t=0).
    Low values  -> consecutive hidden states are nearly identical
                   (representation collapse, often correlates with
                   degenerate / repetitive generation).
    High values -> healthy diversity in representation across steps.
    """
    if len(hidden_states) < 2:
        return []
    scores: List[float] = []
    for t in range(1, len(hidden_states)):
        h_prev = hidden_states[t - 1]
        h_curr = hidden_states[t]
        denom = (np.linalg.norm(h_prev) * np.linalg.norm(h_curr)) + 1e-12
        cos_sim = float(np.dot(h_prev, h_curr) / denom)
        scores.append(1.0 - cos_sim)
    return scores


def aggregate_internal_signals(
    logits_per_token: Optional[List[np.ndarray]],
    hidden_states_per_token: Optional[List[np.ndarray]],
) -> dict:
    """Summarise the per-token internal signals.

    Returns a dict with mean / extremum summaries plus the per-token
    sequences (the latter for charting on the frontend). When either
    input is missing (i.e. API mode) all summary fields are None.
    """
    if logits_per_token is None or hidden_states_per_token is None:
        return {
            "mean_semantic_energy": None,
            "mean_rcd": None,
            "min_rcd": None,
            "per_token_energy": [],
            "per_token_rcd": [],
        }

    energies = semantic_energy(logits_per_token)
    rcds = representation_collapse(hidden_states_per_token)

    return {
        "mean_semantic_energy": round(float(np.mean(energies)), 4) if energies else 0.0,
        "mean_rcd": round(float(np.mean(rcds)), 4) if rcds else 0.0,
        "min_rcd": round(float(np.min(rcds)), 4) if rcds else 0.0,
        "per_token_energy": [round(float(e), 4) for e in energies],
        "per_token_rcd": [round(float(r), 4) for r in rcds],
    }
