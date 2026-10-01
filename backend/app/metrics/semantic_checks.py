"""Backward-compatible shim — see halluscope_core.metrics.semantic_checks."""

from halluscope_core.metrics.semantic_checks import (  # noqa: F401
    semantic_novelty,
    entity_expansion,
    claim_density,
    compute_semantic_checks,
    EmbedFn,
)
