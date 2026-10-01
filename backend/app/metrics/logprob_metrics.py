"""Backward-compatible shim — see halluscope_core.metrics.logprob_metrics."""

from halluscope_core.metrics.logprob_metrics import (  # noqa: F401
    mean_logprob,
    logprob_variance,
    logprob_std,
)
