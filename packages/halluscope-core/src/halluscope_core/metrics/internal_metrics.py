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


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - float(np.max(z))
    e = np.exp(z)
    return e / float(np.sum(e))


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) for two probability vectors."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def logit_lens_divergence_token(
    per_layer_hidden: List[np.ndarray],
    unembed: np.ndarray,
    layer_stride: int = 1,
) -> float:
    """Logit Lens Divergence for ONE token (≥ 0; higher = riskier).

    Each layer's hidden state is projected to the vocabulary via the output
    (unembedding) matrix -- the "logit lens" -- and turned into a distribution.
    LLD is the mean symmetric-free KL between adjacent (strided) layers'
    distributions: it measures how much the model's next-token prediction
    churns as it moves up the layer stack. A stable, confident prediction
    barely changes across depth (low LLD); an unstable one swings (high LLD).

    Args:
        per_layer_hidden: list of (hidden_dim,) vectors, one per layer.
        unembed: (vocab, hidden_dim) projection matrix (lm_head weight).
        layer_stride: compare layers ``i`` and ``i+stride`` (>=1).
    """
    layers = per_layer_hidden[:: max(1, layer_stride)]
    if len(layers) < 2:
        return 0.0
    dists = [_softmax(unembed @ h) for h in layers]
    kls = [_kl(dists[i], dists[i + 1]) for i in range(len(dists) - 1)]
    return float(np.mean(kls)) if kls else 0.0


def semantic_confidence(token_hidden: np.ndarray, context_vector: np.ndarray) -> float:
    """Map cos(token_hidden, context_vector) from [-1, 1] onto [0, 1].

    1.0 => the chosen token's representation is perfectly aligned with the
    pooled context (semantically coherent); 0.0 => anti-aligned.
    """
    denom = (
        float(np.linalg.norm(token_hidden)) * float(np.linalg.norm(context_vector))
    ) + 1e-12
    cos = float(np.dot(token_hidden, context_vector) / denom)
    return (cos + 1.0) / 2.0


def confidence_stability_gap(
    token_top1_prob: float,
    token_hidden: np.ndarray,
    context_vector: np.ndarray,
) -> float:
    """Confidence-Stability Gap for one token (∈ [0, 1]; higher = riskier).

    CSG = |C_semantic - C_token| where both confidences live in [0, 1]:

      * C_token   = the scalar top-1 softmax probability of the chosen token
                    (how locally certain the model was), clamped to [0, 1].
      * C_semantic = semantic_confidence(token_hidden, context_vector), the
                    cosine alignment of the token's hidden state with the
                    mean-pooled context, mapped to [0, 1].

    A large gap means the model was locally confident about a token whose
    representation is semantically out of step with its own context (or vice
    versa) — a hallucination tell.
    """
    c_token = min(1.0, max(0.0, float(token_top1_prob)))
    c_sem = semantic_confidence(token_hidden, context_vector)
    return abs(c_sem - c_token)


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


def mean_logit_lens_divergence(
    per_token_layer_hidden: List[np.ndarray],
    unembed: np.ndarray,
    layer_stride: int = 1,
) -> float:
    """Mean Logit Lens Divergence over a token sequence.

    ``per_token_layer_hidden[i]`` is a (n_layers, hidden_dim) stack for token i.
    """
    if not per_token_layer_hidden:
        return 0.0
    vals = [
        logit_lens_divergence_token(list(layers), unembed, layer_stride)
        for layers in per_token_layer_hidden
    ]
    return float(np.mean(vals)) if vals else 0.0


def mean_confidence_stability_gap(
    token_top1_probs: List[float],
    hidden_states_per_token: List[np.ndarray],
) -> float:
    """Mean Confidence-Stability Gap over a sequence.

    The context vector is the mean-pooled last-layer hidden state across the
    generated tokens; each token's CSG is taken against that context.
    """
    n = min(len(token_top1_probs), len(hidden_states_per_token))
    if n == 0:
        return 0.0
    context = np.mean(np.stack(hidden_states_per_token[:n]), axis=0)
    gaps = [
        confidence_stability_gap(
            token_top1_probs[i], hidden_states_per_token[i], context
        )
        for i in range(n)
    ]
    return float(np.mean(gaps)) if gaps else 0.0


def _top1_probs(logits_per_token: List[np.ndarray]) -> List[float]:
    return [float(np.max(_softmax(z))) for z in logits_per_token]


def aggregate_internal_signals(
    logits_per_token: Optional[List[np.ndarray]],
    hidden_states_per_token: Optional[List[np.ndarray]],
    per_layer_hidden_per_token: Optional[List[np.ndarray]] = None,
    unembed: Optional[np.ndarray] = None,
    layer_stride: int = 1,
) -> dict:
    """Summarise the per-token internal signals.

    Returns mean / extremum summaries plus per-token sequences (for charting).
    When logits/hidden are missing (API mode) every summary field is None.
    ``mean_lld`` is None unless per-layer hidden states AND an unembed matrix
    are supplied (only the local adapter can provide them).
    """
    if logits_per_token is None or hidden_states_per_token is None:
        return {
            "mean_semantic_energy": None,
            "mean_rcd": None,
            "min_rcd": None,
            "mean_lld": None,
            "mean_csg": None,
            "per_token_energy": [],
            "per_token_rcd": [],
        }

    energies = semantic_energy(logits_per_token)
    rcds = representation_collapse(hidden_states_per_token)
    top1 = _top1_probs(logits_per_token)
    mean_csg = mean_confidence_stability_gap(top1, hidden_states_per_token)

    mean_lld = None
    if per_layer_hidden_per_token is not None and unembed is not None:
        mean_lld = round(
            mean_logit_lens_divergence(
                per_layer_hidden_per_token, unembed, layer_stride
            ),
            4,
        )

    return {
        "mean_semantic_energy": round(float(np.mean(energies)), 4) if energies else 0.0,
        "mean_rcd": round(float(np.mean(rcds)), 4) if rcds else 0.0,
        "min_rcd": round(float(np.min(rcds)), 4) if rcds else 0.0,
        "mean_lld": mean_lld,
        "mean_csg": round(float(mean_csg), 4),
        "per_token_energy": [round(float(e), 4) for e in energies],
        "per_token_rcd": [round(float(r), 4) for r in rcds],
    }
