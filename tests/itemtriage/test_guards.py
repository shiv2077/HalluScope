"""The preregistration guard, the unblinding record, and the cost cap."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from itemtriage import costs, evaluate  # noqa: E402


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


def test_prereg_guard_and_unblinding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    monkeypatch.setattr(evaluate, "REPO", repo)
    monkeypatch.setattr(evaluate, "UNBLIND", tmp_path / "unblinding.json")
    (repo / "docs" / "PREREG.md").write_text("v1\n")
    with pytest.raises(SystemExit, match="not tracked"):
        evaluate.prereg_commit()
    _git(repo, "add", "docs/PREREG.md")
    _git(repo, "commit", "-q", "-m", "prereg")
    c1 = evaluate.prereg_commit()
    assert evaluate.unblind(c1) == {"prereg_commit": c1, "unblinded_under": c1, "deviation": False}
    (repo / "docs" / "PREREG.md").write_text("v2\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        evaluate.prereg_commit()
    _git(repo, "commit", "-q", "-am", "edit after unblinding")
    c2 = evaluate.prereg_commit()
    assert c2 != c1
    assert evaluate.unblind(c2)["deviation"] is True
    assert json.loads((tmp_path / "unblinding.json").read_text())["prereg_commit"] == c1


def test_cost_gate_refuses_over_cap():
    over = [{"model": "gpt-4o", "purpose": "mc:test-only", "n_calls": 10_000_000, "est_input_tokens": 1000, "est_output_tokens": 1}]
    with pytest.raises(SystemExit, match="exceeds"):
        costs.gate(over)
    small = [{"model": "gpt-4o-mini", "purpose": "mc:test-only", "n_calls": 1, "est_input_tokens": 100, "est_output_tokens": 1}]
    assert costs.gate(small)["within_cap"]
