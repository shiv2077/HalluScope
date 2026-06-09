"""Probe bank, perturbation construction, and strata assignment. No model."""

from src.hv2.perturb import (
    PERTURB_TYPES,
    build_perturbations,
    p1_wrong_context,
    p2_fewshot_wrong,
    p3_format_pressure,
)
from src.hv2.probes import build_probes, know_band, know_score
from src.hv2.strata import assign_stratum, strata_table


# ---- probes ----
def test_eight_probes_4_greedy_4_temp():
    probes = build_probes("What is the capital of France?")
    assert len(probes) == 8
    greedy = [p for p in probes if p.greedy]
    temp = [p for p in probes if not p.greedy]
    assert len(greedy) == 4 and len(temp) == 4
    assert all(p.temperature == 0.7 for p in temp)
    assert all("France" in p.text for p in probes)  # question embedded in every probe


def test_know_score_and_band():
    assert know_score([True] * 6 + [False] * 2) == 0.75
    assert know_band(6) == "high"
    assert know_band(2) == "low"
    assert know_band(4) == "mid"


# ---- perturbations ----
def test_p1_includes_distractor_and_question():
    txt = p1_wrong_context("Who painted the Mona Lisa?", "Pablo Picasso")
    assert "Pablo Picasso" in txt and "Mona Lisa" in txt


def test_p2_corrupts_two_exemplars():
    exemplars = [("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]
    txt = p2_fewshot_wrong("target?", exemplars, ["WRONG_A", "WRONG_B"])
    assert "WRONG_A" in txt and "WRONG_B" in txt
    assert "a1" in txt and "a3" in txt  # positions 0,2 stay correct
    assert "a2" not in txt and "a4" not in txt  # positions 1,3 corrupted


def test_p3_demands_answer():
    txt = p3_format_pressure("Capital of Peru?")
    assert "must answer" in txt.lower()


def test_build_perturbations_three_types():
    perts = build_perturbations(
        "Q?", "Distractor", [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")], ["W1", "W2"]
    )
    assert [p.perturb_type for p in perts] == list(PERTURB_TYPES)


# ---- strata ----
def test_stratum_labels():
    assert assign_stratum(7, generation_correct=False) == "KH"  # knows but wrong
    assert assign_stratum(8, generation_correct=True) == "KC"
    assert assign_stratum(1, generation_correct=False) == "UH"
    assert assign_stratum(0, generation_correct=True) == "UC"
    assert assign_stratum(4, generation_correct=False) == "MID"


def test_strata_table_breakdown_by_perturbation():
    records = [
        {"n_correct_probes": 7, "correct": False, "perturb_type": "P1"},  # KH
        {"n_correct_probes": 7, "correct": True, "perturb_type": "P1"},  # KC
        {"n_correct_probes": 1, "correct": False, "perturb_type": "P2"},  # UH
        {"n_correct_probes": 8, "correct": False, "perturb_type": "P3"},  # KH
    ]
    t = strata_table(records)
    assert t["overall"]["KH"] == 2
    assert t["by_perturbation"]["P1"]["KH"] == 1
    assert t["by_perturbation"]["P3"]["KH"] == 1
    assert t["n"] == 4
