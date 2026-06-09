"""Threshold unification + config-driven HRI behaviour.

Historically the HIGH cutoff was 0.30 in hri_engine, 0.46 in
intervention_service, and 0.086 in eval artifacts. These tests pin a single
source of truth: classification and intervention must use the SAME threshold,
read from config/thresholds.json.
"""

from __future__ import annotations

from backend.app import config_store
from backend.app.risk.hri_engine import classify_risk, compute_hri
from backend.app.services.intervention_service import should_intervene


def test_classification_and_intervention_share_one_threshold():
    high = config_store.risk_high()
    # Exactly at the HIGH cutoff: HIGH band AND intervention fires.
    assert classify_risk(high) == "HIGH"
    assert should_intervene(high) is True
    # Just below: not HIGH, and must NOT intervene (no 0.30/0.46 gap).
    just_below = high - 1e-6
    assert classify_risk(just_below) != "HIGH"
    assert should_intervene(just_below) is False


def test_classify_risk_bands():
    low = config_store.risk_low()
    high = config_store.risk_high()
    assert classify_risk(low - 1e-6) == "LOW"
    assert classify_risk((low + high) / 2.0) == "MEDIUM"
    assert classify_risk(high + 1e-6) == "HIGH"


def test_higher_semantic_energy_raises_hri():
    """With logprobs held constant, a flatter (higher-energy) internal signal
    must produce a higher HRI — proves the energy term points the right way
    after the direction fix and through the config-driven weighted sum."""
    logprobs = [-0.5, -0.6, -0.4, -0.55, -0.5, -0.6]
    top = [[-0.5, -1.0, -2.0, -3.0, -4.0]] * len(logprobs)

    low_energy = {"mean_semantic_energy": 0.10, "mean_rcd": 0.3}
    high_energy = {"mean_semantic_energy": 0.90, "mean_rcd": 0.3}

    hri_low = compute_hri(logprobs, top, internal_signals=low_energy)
    hri_high = compute_hri(logprobs, top, internal_signals=high_energy)

    assert hri_low["hri_mode"] == "local_internal"
    assert hri_high["hri"] > hri_low["hri"], (
        f"flatter logits (energy 0.9) must raise HRI vs energy 0.1: "
        f"{hri_high['hri']} !> {hri_low['hri']}"
    )


def test_api_mode_used_when_no_internal_signals():
    logprobs = [-0.5, -0.6, -0.4, -0.55]
    res = compute_hri(logprobs)
    assert res["hri_mode"] == "api"
