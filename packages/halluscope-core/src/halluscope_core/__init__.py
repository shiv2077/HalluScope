"""halluscope-core — the HRI scoring engine, with no FastAPI/SQLite/React dependency.

Usage::

    from halluscope_core.risk.hri_engine import compute_hri

    result = compute_hri(logprobs=[-0.05, -0.12, -0.08, -1.40, -0.30])
    print(result["hri"], result["risk_level"])
    # 0.0182 LOW

This package is the scoring core factored out of the HalluScope reference app
(see the repo root ``README.md`` and ``docs/DECISIONS.md`` ADR-003/ADR-013 for
why the app keeps two feature sets behind one interface). It reads
configuration from the monorepo's ``config/*.json`` by default; set
``HALLUSCOPE_CONFIG_DIR`` to point elsewhere when using it standalone.
"""

from halluscope_core.risk.hri_engine import (
    compute_hri,
    hri_from_features,
    classify_risk,
    FEATURE_NAMES_API,
    FEATURE_NAMES_LOCAL,
)

__all__ = [
    "compute_hri",
    "hri_from_features",
    "classify_risk",
    "FEATURE_NAMES_API",
    "FEATURE_NAMES_LOCAL",
]

__version__ = "0.1.0"
