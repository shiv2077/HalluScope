"""Tests for the deterministic, disjoint train/test split builder."""

from __future__ import annotations

from eval.build_subset import build_split, normalize_row


def _rows(n):
    # Synthetic dataset rows with some duplicate questions.
    out = []
    for i in range(n):
        q = f"Question {i % (n - 5)}?"  # force a few duplicates
        out.append({"question": q, "right_answer": f"ans {i}"})
    return out


def test_split_is_disjoint_and_sized():
    train, test = build_split(_rows(500), n_train=100, n_test=50, seed=42)
    assert len(train) == 100
    assert len(test) == 50
    tq = {r["question"] for r in train}
    sq = {r["question"] for r in test}
    assert tq.isdisjoint(sq)  # no leakage
    assert len(tq) == 100  # all-unique within train
    assert len(sq) == 50


def test_split_is_deterministic():
    a = build_split(_rows(500), 80, 40, seed=7)
    b = build_split(_rows(500), 80, 40, seed=7)
    assert [r["id"] for r in a[0]] == [r["id"] for r in b[0]]
    assert [r["id"] for r in a[1]] == [r["id"] for r in b[1]]


def test_split_different_seeds_differ():
    a = build_split(_rows(500), 80, 40, seed=1)
    b = build_split(_rows(500), 80, 40, seed=2)
    assert [r["question"] for r in a[0]] != [r["question"] for r in b[0]]


def test_normalize_row_halueval_and_truthfulqa():
    h = normalize_row({"question": " Q? ", "right_answer": " A "}, "halueval_qa")
    assert h["question"] == "Q?" and h["reference_answer"] == "A"
    t = normalize_row(
        {
            "question": "Q2?",
            "best_answer": "Best",
            "correct_answers": ["Best", "Also"],
            "incorrect_answers": ["No"],
        },
        "truthfulqa",
    )
    assert t["reference_answer"] == "Best"
    assert t["correct_answers"] == ["Best", "Also"]
