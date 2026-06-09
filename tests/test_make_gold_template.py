"""Tests for the pure pieces of make_gold_template: disjoint selection and the
no-invent answer fill."""

from __future__ import annotations

from eval.make_gold_template import select_gold, fill_model_answers


def _pool(n=200):
    return [{"question": f"Q{i}?", "right_answer": f"A{i}"} for i in range(n)]


def test_select_is_disjoint_from_excluded():
    exclude = {f"Q{i}?" for i in range(50)}  # first 50 questions reserved
    picked = select_gold(_pool(), n=80, seed=42, exclude_questions=exclude)
    assert len(picked) == 80
    assert all(p["question"] not in exclude for p in picked)
    assert len({p["question"] for p in picked}) == 80  # unique
    assert all(p["id"].startswith("gold_") for p in picked)


def test_select_is_deterministic():
    a = select_gold(_pool(), 40, seed=7, exclude_questions=set())
    b = select_gold(_pool(), 40, seed=7, exclude_questions=set())
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert [p["question"] for p in a] == [p["question"] for p in b]


def test_fill_never_invents_or_overwrites():
    rows = [
        {"id": "gold_00001", "model_answer": ""},
        {"id": "gold_00002", "model_answer": "already here"},
        {"id": "gold_00003", "model_answer": ""},
    ]
    answers = {"gold_00001": "real answer one"}  # gold_00003 absent
    filled = fill_model_answers(rows, answers)
    assert filled == 1
    assert rows[0]["model_answer"] == "real answer one"  # filled from real
    assert rows[1]["model_answer"] == "already here"  # not overwritten
    assert rows[2]["model_answer"] == ""  # no invention
