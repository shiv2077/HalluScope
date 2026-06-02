"""Centralised loader for the JSON config files under ``<repo>/config``.

No magic numbers live in the engine code. Risk thresholds, HRI weights, and
normalization constants are all defined in ``config/*.json`` and read through
this module so that:

  * ``hri_engine`` and ``intervention_service`` agree on the SAME threshold
    (one source of truth — fixes the historical 0.30 / 0.46 / 0.086 drift);
  * ``eval/fit_weights.py`` can overwrite ``config/weights.json`` with learned
    coefficients and the engine picks them up with no code change.

Values are cached after first read. Call :func:`reload` to drop the cache
(used by tests and by tools that have just rewritten a config file).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

# backend/app/config_store.py -> backend/app -> backend -> repo root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"


@lru_cache(maxsize=None)
def _load(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing config file {path}. Expected JSON under {CONFIG_DIR}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def thresholds() -> Dict[str, Any]:
    """Risk-band thresholds (``risk_low``, ``risk_high``)."""
    return _load("thresholds.json")


def weights() -> Dict[str, Any]:
    """HRI combination weights + sigmoid params (api / local / prompt_risk)."""
    return _load("weights.json")


def norm() -> Dict[str, Any]:
    """Normalization constants for raw signals."""
    return _load("norm.json")


def risk_high() -> float:
    return float(thresholds()["risk_high"])


def risk_low() -> float:
    return float(thresholds()["risk_low"])


def reload() -> None:
    """Clear the cache so the next access re-reads from disk."""
    _load.cache_clear()
