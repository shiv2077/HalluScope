"""flag_by_budget flags exactly ceil(budget * n) rows and is deterministic across input order."""

from __future__ import annotations

import builtins
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from routing.policy import batch_cutoff, flag_by_budget, flag_by_risk  # noqa: E402


@pytest.mark.parametrize("budget", [0.10, 0.20, 0.30])
@pytest.mark.parametrize("n", [50, 288, 301])
def test_exact_count_and_determinism(budget, n):
    rng = np.random.RandomState(n)
    qids = [f"q{i:04d}" for i in range(n)]
    scores = pd.Series(np.round(rng.rand(n), 2), index=qids)  # rounding creates ties
    fl = flag_by_budget(scores, budget)
    assert fl.sum() == math.ceil(budget * n)
    for seed in range(3):
        perm = rng.permutation(n)
        fl2 = flag_by_budget(scores.iloc[perm], budget)
        assert set(fl2[fl2].index) == set(fl[fl].index)


def test_ties_by_qid_and_nan():
    scores = pd.Series([1.0, 1.0, 1.0, float("nan"), 0.5], index=["c", "a", "b", "z", "d"])
    fl = flag_by_budget(scores, 0.4)  # ceil(2.0) = 2
    assert list(fl[fl].index) == ["a", "b"]
    assert not fl["z"]
    assert batch_cutoff(scores, 0.4) == 1.0
    assert flag_by_risk(scores, 0.75).tolist() == [True, True, True, False, False]


def test_budget_bounds():
    s = pd.Series([1.0, 2.0])
    for b in (0, 1, -0.1, 1.5):
        with pytest.raises(AssertionError):
            flag_by_budget(s, b)


def test_route_budget_never_opens_calibration(tmp_path, monkeypatch):
    """--policy budget must not open results/calibration.json."""
    import json

    from routing import cli

    rows = [{"qid": f"q{i}", "answer_text": "a", "token_logprobs": json.dumps([-0.1 * (i % 7), -0.5])} for i in range(23)]
    src = tmp_path / "answers.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    out = tmp_path / "routed.csv"
    real_open = builtins.open

    def guard(file, *a, **k):
        if "calibration.json" in str(file):
            raise AssertionError("budget policy opened calibration.json")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guard)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: guard(self) and None if "calibration.json" in str(self) else real_open(self, *a, **k).read())
    cli.main(["route", "--policy", "budget", "--scorer", "mean_logprob", "--budget", "0.20",
              "--in", str(src), "--out", str(out)])
    df = pd.read_csv(out)
    assert (df["lane"] == "review").sum() == math.ceil(0.20 * 23)
    assert set(df["policy"]) == {"budget"} and df["cutoff_applied"].nunique() == 1
