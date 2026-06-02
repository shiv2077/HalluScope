"""
HalluScope – Adapter Schemas

Defines the standardised data structures returned by every adapter.
All adapters must return a StandardizedResult so that the metrics
and risk engine remain model-agnostic.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class StandardizedResult:
    """Unified result object returned by every LLM adapter.

    Attributes:
        text:                    The full generated text.
        token_logprobs:          Per-token log probabilities (natural log).
        token_strings:           The actual token strings (for heatmap display).
        top_logprobs:            For each token position, the log-probs of the
                                 top-K alternative tokens (optional; depends on
                                 provider support).
        model_name:              Identifier of the model that produced the output.
        hidden_states_per_token: Last-layer hidden state vector for each
                                 generated token. Shape per element: (hidden_dim,).
                                 Only populated by adapters that have access
                                 to model internals (i.e. local, not API).
        logits_per_token:        Full pre-softmax logit vector per generated
                                 token. Shape per element: (vocab_size,).
                                 Local-only; never persisted (memory cost).
        per_layer_hidden_per_token: Per-token stack of EVERY layer's hidden
                                 state for the generated token. Shape per
                                 element: (n_layers, hidden_dim). Needed for
                                 Logit Lens Divergence. Local-only and gated by
                                 the HALLUSCOPE_CAPTURE_PER_LAYER env flag
                                 (memory cost); None when not captured.
        unembed:                 Output (unembedding) projection matrix,
                                 shape (vocab_size, hidden_dim), used to apply
                                 the logit lens to per-layer hidden states.
                                 Local-only; shared reference, not persisted.
    """

    text: str
    token_logprobs: List[float]
    token_strings: List[str] = field(default_factory=list)
    top_logprobs: Optional[List[List[float]]] = field(default=None)
    model_name: str = ""
    hidden_states_per_token: Optional[List[np.ndarray]] = field(default=None)
    logits_per_token: Optional[List[np.ndarray]] = field(default=None)
    per_layer_hidden_per_token: Optional[List[np.ndarray]] = field(default=None)
    unembed: Optional[np.ndarray] = field(default=None)
