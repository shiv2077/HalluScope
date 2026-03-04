"""
HalluScope – Adapter Schemas

Defines the standardised data structures returned by every adapter.
All adapters must return a StandardizedResult so that the metrics
and risk engine remain model-agnostic.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StandardizedResult:
    """Unified result object returned by every LLM adapter.

    Attributes:
        text:           The full generated text.
        token_logprobs: Per-token log probabilities (natural log).
        token_strings:  The actual token strings (for heatmap display).
        top_logprobs:   For each token position, the log-probs of the
                        top-K alternative tokens (optional; depends on
                        provider support).
        model_name:     Identifier of the model that produced the output.
    """

    text: str
    token_logprobs: List[float]
    token_strings: List[str] = field(default_factory=list)
    top_logprobs: Optional[List[List[float]]] = field(default=None)
    model_name: str = ""
