"""Backward-compatible shim — see halluscope_core.risk.hri_engine.

The HRI engine now lives in the standalone ``halluscope-core`` package so it
can be installed and imported without FastAPI/SQLite (see docs/DECISIONS.md
and docs/SYSTEM_DESIGN.md). Re-exported here, unchanged, so existing
``from backend.app.risk.hri_engine import ...`` call sites across the app,
eval/, and tests/ keep working with no changes.
"""

from halluscope_core.risk.hri_engine import (  # noqa: F401
    compute_hri,
    hri_from_features,
    classify_risk,
    FEATURE_NAMES_API,
    FEATURE_NAMES_LOCAL,
)
