"""Backward-compatible shim — see halluscope_core.metrics.nlp_backend."""

from halluscope_core.metrics.nlp_backend import (  # noqa: F401
    extract_entities,
    rare_entity_fraction,
    entity_set,
)
