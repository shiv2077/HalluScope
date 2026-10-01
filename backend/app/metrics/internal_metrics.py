"""Backward-compatible shim — see halluscope_core.metrics.internal_metrics."""

from halluscope_core.metrics.internal_metrics import (  # noqa: F401
    semantic_energy,
    free_energy,
    logit_lens_divergence_token,
    semantic_confidence,
    confidence_stability_gap,
    representation_collapse,
    mean_logit_lens_divergence,
    mean_confidence_stability_gap,
    aggregate_internal_signals,
)
