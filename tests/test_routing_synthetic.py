"""End-to-end check of the routing pipeline on synthetic files in a temp dir.

Synthetic data never touches data/ or results/: the test points the package at a temp dir through
ROUTING_*_DIR environment variables before importing it. It verifies the plumbing (contract, scorers,
gates, calibration, evaluation, report) and the calibrate/route logic; it says nothing about real lift.
Run: python -m pytest tests/test_routing_synthetic.py -q
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make(tmp: Path, n_cal=60, n_test=90, seed=0):
    rng = np.random.RandomState(seed)
    for split, n, prefix in (("calibration", n_cal, "cal"), ("test", n_test, "tst")):
        qids = [f"{prefix}_{i:04d}" for i in range(n)]
        pd.DataFrame({"qid": qids, "question": [f"q {q}" for q in qids], "reference_answer": ["a"] * n}) \
            .to_csv(tmp / f"{split}_questions.csv", index=False)
        gens, labels = [], []
        for q in qids:
            wrong = rng.rand() < 0.4
            length = int(rng.randint(5, 40))
            lps = list(np.round(rng.normal(-1.2 if wrong else -0.3, 0.4, length), 4))
            base = dict(qid=q, input_tokens=30, output_tokens=length, temperature=0.0, seed=42)
            gens.append({**base, "model": "gpt-4o-mini", "role": "primary", "answer_text": f"answer {q} {'x' if wrong else 'y'}",
                         "token_logprobs": json.dumps(lps)})
            for k in (1, 2, 3):
                gens.append({**base, "model": "gpt-4o-mini", "role": f"resample_{k}", "temperature": 0.7, "seed": 42 + k,
                             "answer_text": f"answer {q} {'z' + str(k) if wrong else 'y'}", "token_logprobs": "[]"})
            gens.append({**base, "model": "gpt-4o", "role": "strong", "answer_text": "strong", "token_logprobs": "[]"})
            pyes = 0.3 if wrong else 0.9
            gens.append({**base, "model": "gpt-4o-mini", "role": "ptrue", "output_tokens": 1, "answer_text": "Yes",
                         "token_logprobs": json.dumps([{"token": "Yes", "logprob": float(np.log(pyes))},
                                                       {"token": "No", "logprob": float(np.log(1 - pyes))}])})
            for j in ("judge_a", "judge_b"):
                v = "INCORRECT" if wrong else "CORRECT"
                if rng.rand() < 0.05:
                    v = "AMBIGUOUS"
                labels.append(dict(qid=q, model="gpt-4o-mini", judge=j, verdict=v, reason="r"))
                labels.append(dict(qid=q, model="gpt-4o", judge=j, verdict="INCORRECT" if rng.rand() < 0.15 else "CORRECT", reason="r"))
        pd.DataFrame(gens).to_csv(tmp / f"{split}_generations.csv", index=False)
        pd.DataFrame(labels).to_csv(tmp / f"{split}_labels.csv", index=False)


def test_pipeline(tmp_path):
    for k in ("DATA", "RESULTS", "FIGURES", "REPORTS"):
        os.environ[f"ROUTING_{k}_DIR"] = str(tmp_path / k.lower())
        (tmp_path / k.lower()).mkdir()
    for m in list(sys.modules):
        if m.startswith("routing"):
            del sys.modules[m]
    data = importlib.import_module("routing.data")
    _make(tmp_path / "data")
    d = data.load_split("calibration")
    assert len(d["consensus"]) == 120
    scorers = importlib.import_module("routing.scorers")
    scorers.score_split("calibration")
    scorers.score_split("test", scorers.load_coef())
    s = data.read_scores("test")
    assert set(data.SCORERS) <= set(s.columns)
    assert s["mean_logprob"].notna().all() and s["p_true"].between(0, 1).all()
    gates = importlib.import_module("routing.gates")
    g = gates.run("test")
    assert g["template_rows"] == 0 and g["scorers"]["mean_logprob"]["auroc"]["est"] > 0.8
    calibrate = importlib.import_module("routing.calibrate")
    c = calibrate.run()
    cut = c["budgets"]["mean_logprob"]["0.2"]
    cal = data.read_scores("calibration")
    frac = (cal["mean_logprob"] >= cut).mean()
    assert abs(frac - 0.2) < 0.03
    r = c["risk"]["mean_logprob"]["0.2"]
    assert r["threshold"] is None or r["ucb"] <= 0.2
    evaluate = importlib.import_module("routing.evaluate")
    ev = evaluate.run()
    rep = (tmp_path / "reports" / "routing_report.md").read_text()
    assert "## Summary" in rep and "[" in rep
    for f in ("review_lift.png", "cascade_frontier.png", "risk_coverage.png"):
        assert (tmp_path / "figures" / f).exists()
    assert ev["review_lift"]["mean_logprob"]["0.2"]["lift"]["est"] > 1.0
    # determinism: rerun and compare bytes
    ev2 = evaluate.run()
    assert (tmp_path / "reports" / "routing_report.md").read_text() == rep
    # disjointness assertion fires
    q = pd.read_csv(tmp_path / "data" / "test_questions.csv")
    q.loc[0, "qid"] = "cal_0000"
    q.to_csv(tmp_path / "data" / "test_questions.csv", index=False)
    try:
        data.assert_disjoint()
        raise AssertionError("disjointness assertion did not fire")
    except AssertionError as e:
        assert "share" in str(e)
