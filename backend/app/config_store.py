"""Backward-compatible shim.

The config loader now lives in the standalone ``halluscope-core`` package
(``packages/halluscope-core/src/halluscope_core/config_store.py``) so it can
be installed and imported without FastAPI/SQLite (see docs/DECISIONS.md and
docs/SYSTEM_DESIGN.md). Re-exported here, unchanged, so existing
``from backend.app import config_store`` call sites across the app, eval/,
and tests/ keep working with no changes.
"""

from halluscope_core.config_store import (  # noqa: F401
    CONFIG_DIR,
    ROOT_DIR,
    thresholds,
    weights,
    norm,
    grader,
    prompt_risk,
    risk_high,
    risk_low,
    reload,
)
