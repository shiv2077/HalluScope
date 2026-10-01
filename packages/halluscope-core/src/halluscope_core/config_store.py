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
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

# packages/halluscope-core/src/halluscope_core/config_store.py -> ... -> repo root.
# HalluScope is a monorepo (see docs/DECISIONS.md ADR-013 framing): this package is
# installable on its own, but its default config directory is the monorepo's
# config/*.json — the single source of truth shared with the FastAPI app and eval/.
# Set HALLUSCOPE_CONFIG_DIR to point at a different directory (same JSON schema) when
# using this package outside the monorepo.
_REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT_DIR = _REPO_ROOT
CONFIG_DIR = Path(os.environ.get("HALLUSCOPE_CONFIG_DIR", str(_REPO_ROOT / "config")))


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


def grader() -> Dict[str, Any]:
    """SBERT answer-grading config (model + cosine threshold)."""
    return _load("grader.json")


def prompt_risk() -> Dict[str, Any]:
    """Stage 1 prompt-risk parameters (cutoff year, relative markers)."""
    return _load("prompt_risk.json")


def risk_high() -> float:
    return float(thresholds()["risk_high"])


def risk_low() -> float:
    return float(thresholds()["risk_low"])


def reload() -> None:
    """Clear the cache so the next access re-reads from disk."""
    _load.cache_clear()
