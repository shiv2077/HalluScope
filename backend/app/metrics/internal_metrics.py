"""Internal-signal metrics for local-mode HRI.

These metrics require access to model internals (full pre-softmax logits
and last-layer hidden states) and therefore cannot be computed for
API-based providers like OpenAI.

Implemented:
  - Semantic Energy (flatness): F_t = (logsumexp(z_t) - max(z_t)) / log(V)
  - Representation Collapse:     RCD_t = 1 - cos(h_t, h_{t-1})

Semantic-energy direction fix (see tests/test_internal_metrics.py)
------------------------------------------------------------------
The original term used the raw free energy E_t = -logsumexp(z_t) and the
HRI engine consumed it as abs(E_t)/35. That had three defects:

  1. Unbounded — abs(E)/35 only *happened* to land near [0, 1] for one
     vocab size; for any other model it drifts.
  2. Magnitude-dominated — logsumexp(z) ≈ max(z), so E tracked the size of
     the largest logit, not the shape of the distribution. Adding a constant
     c to every logit (which leaves the softmax unchanged) shifted E by c.
  3. Inverted — a *more confident* (peaked) token has a *larger* max logit,
     hence larger logsumexp, hence larger |E|, hence HIGHER risk. Backwards.

The fix keeps the log-sum-exp / free-energy framing but measures the
shift-invariant log-partition mass *relative to the mode*:

    F_t = logsumexp(z_t) - max(z_t)
        = log( sum_i exp(z_t,i - max_i) )            # in [0, log V]
        = -log( softmax(z_t)_argmax )                # surprisal of the top token

F_t is 0 when one token dominates (peaked / confident) and log(V) when the
distribution is uniform (maximally flat / uncertain). Dividing by log(V)
gives a bounded flatness energy in [0, 1] that is invariant to a global
logit shift and rises monotonically as the distribution flattens.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


def semantic_energy(logits_per_token: List[np.ndarray]) -> List[float]:
    """Per-token semantic (flatness) energy in [0, 1].

        F_t = (logsumexp(z_t) - max(z_t)) / log(V)

    Higher = flatter logits = more uncertainty = more risk. Bounded to
    [0, 1] and invariant to a global shift of the logits (which does not
    change the softmax). Uses the log-sum-exp trick for numerical stability.
    """
    energies: List[float] = []
    for z in logits_per_token:
        vocab = int(z.shape[-1])
        if vocab <= 1:
            energies.append(0.0)
            continue
        z_max = float(np.max(z))
        # log sum exp(z - z_max) is in [0, log V]; it is exactly the
        # negative log-probability of the most likely token.
        log_partition_rel_mode = math.log(float(np.sum(np.exp(z - z_max))))
        flatness = log_partition_rel_mode / math.log(vocab)
        # Clamp to guard against fp noise at the [0, 1] boundaries.
        energies.append(float(min(1.0, max(0.0, flatness))))
    return energies


def free_energy(logits_per_token: List[np.ndarray]) -> List[float]:
    """Raw OOD-style free energy E_t = -logsumexp(z_t).

    Provided for completeness / OOD-detection use. NOT used by the HRI
    flatness term: it is magnitude-dominated and shift-variant, so it is
    unsuitable as a bounded uncertainty signal (see module docstring).
    """
    out: List[float] = []
    for z in logits_per_token:
        z_max = float(np.max(z))
        lse = z_max + math.log(float(np.sum(np.exp(z - z_max))))
        out.append(float(-lse))
    return out


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
