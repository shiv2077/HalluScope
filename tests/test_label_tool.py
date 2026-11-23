"""The labeling prompt shows only question, reference and candidate; never a score or a verdict."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from routing import data  # noqa: E402
from routing.label_tool import FORBIDDEN, leaks, render  # noqa: E402


def test_render_is_blind():
    text = render("Who wrote Hamlet?", "William Shakespeare", "Shakespeare wrote it.")
    assert leaks(text) == []
    for w in list(data.SCORERS) + list(data.VERDICTS) + ["judge_a", "judge_b", "score"]:
        assert w not in text
    assert "Question: Who wrote Hamlet?" in text and "Candidate answer: Shakespeare wrote it." in text


def test_render_refuses_leak():
    with pytest.raises(AssertionError):
        render("q", "r", "candidate CORRECT per judge_a")
    assert "mean_logprob" in FORBIDDEN


@pytest.mark.skipif(not (ROOT / "data" / "test_generations.csv").exists(), reason="needs committed data")
def test_run_in_memory(tmp_path, monkeypatch):
    """Drive the tool with piped input on the real split, writing to a temp results dir."""
    from routing import label_tool

    monkeypatch.setattr(label_tool, "human_path", lambda split: tmp_path / f"{split}_human_labels.csv")
    monkeypatch.setattr(label_tool, "sample_path", lambda split, seed: tmp_path / f"sample_{split}_{seed}.json")
    out = io.StringIO()
    n = label_tool.run("test", 3, 123, "pytest", stdin=io.StringIO("c\ni a note\nx\na\n"), stdout=out)
    assert n == 3
    assert leaks(out.getvalue()) == []
    rows = (tmp_path / "test_human_labels.csv").read_text().splitlines()
    assert rows[0] == "qid,annotator,verdict,note,timestamp" and len(rows) == 4
    assert "INCORRECT,a note" in rows[2]
    # resumable: second run has nothing to do
    assert label_tool.run("test", 3, 123, "pytest", stdin=io.StringIO(""), stdout=io.StringIO()) == 0
