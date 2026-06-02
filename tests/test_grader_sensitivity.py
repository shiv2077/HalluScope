"""Tests for the (pure) SBERT-threshold sweep used by grader sensitivity.

These cover only the counting logic — no SBERT model is loaded — so the
match/mismatch split arithmetic is verified deterministically.
"""

from __future__ import annotations

from eval.grader_sensitivity import sweep_counts


def test_sweep_counts_split_at_threshold():
    sims = [0.2, 0.4, 0.6, 0.8]
    (row,) = sweep_counts(sims, [0.5])
    assert row["threshold"] == 0.5
    assert row["n_match"] == 2          # 0.6, 0.8 >= 0.5
    assert row["n_mismatch"] == 2       # 0.2, 0.4 < 0.5  (positive = mismatch)
    assert row["n_total"] == 4
    assert row["mismatch_rate"] == 0.5  # positive (hallucination) class rate


def test_sweep_counts_monotonic_in_threshold():
    sims = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = sweep_counts(sims, [0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    # Raising the match threshold can only INCREASE the mismatch (positive) count.
    mismatches = [r["n_mismatch"] for r in rows]
    assert mismatches == sorted(mismatches)


def test_sweep_counts_boundary_is_inclusive_for_match():
    # s == threshold counts as a match (>=), consistent with eval/match.py.
    (row,) = sweep_counts([0.6], [0.6])
    assert row["n_match"] == 1
    assert row["n_mismatch"] == 0
