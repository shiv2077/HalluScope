"""Backward-compatible shim — see halluscope_core.metrics.prompt_risk."""

from halluscope_core.metrics.prompt_risk import (  # noqa: F401
    entity_rarity_signal,
    time_risk,
    prompt_perplexity,
    prompt_perplexity_signal,
    quoted_title_risk,
    compute_prompt_risk,
)
