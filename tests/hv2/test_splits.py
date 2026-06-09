"""Frozen, deterministic, question-disjoint splits. No model/network."""

import json

import pytest

from src.hv2.splits import assign_split, freeze_splits, make_splits, splits_payload


def test_assignment_is_deterministic():
    assert assign_split("tqa_00001") == assign_split("tqa_00001")


def test_splits_are_disjoint_by_question():
    qids = [f"q{i}" for i in range(5000)]
    mapping = make_splits(qids)
    # every qid lands in exactly one split (dict guarantees uniqueness)
    assert set(mapping.values()) <= {"calib", "val", "test"}
    counts = splits_payload(qids)["counts"]
    assert sum(counts.values()) == 5000


def test_fraction_approximately_respected():
    qids = [f"q{i}" for i in range(10000)]
    counts = splits_payload(qids, (0.4, 0.3, 0.3))["counts"]
    assert abs(counts["calib"] / 10000 - 0.4) < 0.03
    assert abs(counts["val"] / 10000 - 0.3) < 0.03
    assert abs(counts["test"] / 10000 - 0.3) < 0.03


def test_freeze_then_reverify_is_idempotent(tmp_path):
    qids = [f"q{i}" for i in range(100)]
    p = tmp_path / "splits.json"
    first = freeze_splits(qids, p)
    second = freeze_splits(qids, p)  # same input -> verifies, no error
    assert first["content_hash"] == second["content_hash"]


def test_freeze_refuses_modification(tmp_path):
    p = tmp_path / "splits.json"
    freeze_splits([f"q{i}" for i in range(100)], p)
    with pytest.raises(RuntimeError, match="frozen splits"):
        freeze_splits([f"q{i}" for i in range(101)], p)  # different set -> refuse


def test_growth_preserves_existing_assignment():
    # adding new questions must not move an existing qid
    small = make_splits([f"q{i}" for i in range(100)])
    big = make_splits([f"q{i}" for i in range(200)])
    for i in range(100):
        assert small[f"q{i}"] == big[f"q{i}"]
