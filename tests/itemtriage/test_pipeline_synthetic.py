"""End-to-end check on synthetic items in temp dirs: split, score, evaluate, calibrate, report.

Never touches data/itemtriage or the real test fold. Flawed items make strong models avoid the key, so a
working pipeline must rank them above chance. Says nothing about real results.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def it(tmp_path, monkeypatch):
    for k in ("DATA", "RESULTS", "FIGURES", "REPORTS"):
        monkeypatch.setenv(f"ITEMTRIAGE_{k}_DIR", str(tmp_path / k.lower()))
        (tmp_path / k.lower()).mkdir()
    saved = {m: v for m, v in sys.modules.items() if m == "itemtriage" or m.startswith("itemtriage.")}
    for m in saved:
        del sys.modules[m]
    mods = {n: importlib.import_module(f"itemtriage.{n}") for n in
            ("paths", "splits", "scorers", "evaluate", "calibrate", "report", "collect")}
    yield mods, tmp_path
    for m in [m for m in sys.modules if m == "itemtriage" or m.startswith("itemtriage.")]:
        del sys.modules[m]
    sys.modules.update(saved)


def make(paths, collect, rng):
    subjects = [f"subj_{i:02d}" for i in range(27)]
    rows, labs, panel = [], [], []
    models = [f"m{j}" for j in range(5)]
    theta = np.array([-1.0, -0.4, 0.1, 0.6, 1.2])
    for s in subjects:
        rate = rng.uniform(0.02, 0.15)
        for i in range(40):
            iid = f"redux:{s}:{i:03d}"
            key = int(rng.randint(4))
            flaw = rng.rand() < rate
            rows.append({"item_id": iid, "source": "synthetic", "subject": s, "question": f"Q {iid}?",
                         "options": [f"{iid} opt {k}" for k in range(4)], "n_options": 4, "key_index": key,
                         "text_hash": iid})
            et = "wrong_groundtruth" if flaw else "ok"
            labs.append({"item_id": iid, "error_type_raw": et, "flaw_primary": int(flaw), "flaw_any": int(flaw),
                         "correct_answer_raw": "", "potential_reason": "", "label_source": "synthetic"})
            other = (key + 1) % 4
            for m, t in zip(models, theta):
                target = other if flaw else key
                strength = 1 / (1 + np.exp(-(t + 0.5)))
                p = np.full(4, (1 - strength) / 3)
                p[target] = strength
                p = rng.dirichlet(p * 20 + 0.1)
                panel.append({"item_id": iid, "model": m, "family": f"F{m}", "access": "local", "p_options": p.tolist(),
                              "p_key": float(p[key]), "argmax_index": int(np.argmax(p)), "parse_ok": True,
                              "letters_missing": 0, "n_orders": 2, "orders_agree": True})
    pd.DataFrame(rows).to_parquet(paths.ITEMS, index=False)
    pd.DataFrame(labs).astype({"flaw_primary": "Int64", "flaw_any": "Int64"}).to_parquet(paths.LABELS, index=False)
    pd.DataFrame(panel).to_parquet(collect.PANEL_OUT, index=False)
    pd.DataFrame(columns=["item_id", "model", "logodds_no_yes", "yes_absent", "no_absent"]).to_parquet(collect.PTRUE_OUT, index=False)
    (paths.RESULTS / "panel.json").write_text(json.dumps({"panel": models}))


def test_pipeline_end_to_end(it, monkeypatch):
    m, tmp = it
    make(m["paths"], m["collect"], np.random.RandomState(0))
    m["splits"].freeze()
    m["scorers"].run()
    sc = pd.read_parquet(m["scorers"].SCORES)
    assert sc["ptrue_key"].isna().all() and sc["conf_w"].notna().all()
    ev = m["evaluate"]
    monkeypatch.setattr(ev, "N_BOOT", 40)
    monkeypatch.setattr(ev, "prereg_commit", lambda: "c" * 40)
    out = ev.run()
    assert "ptrue_key" not in out["scorers_evaluated"]
    assert out["amendments"], "a local-only panel must carry the amendment note"
    p = out["classes"]["primary"]["scorers"]["conf_w"]["precision_0.1"]
    assert p["est"] > out["classes"]["primary"]["base_rate"]["est"]
    assert (m["paths"].RESULTS / "unblinding.json").exists()
    m["calibrate"].run()
    m["report"].write_final()
    text = (m["paths"].REPORTS / "itemtriage_report.md").read_text()
    assert "Amendment 1" in text and "## Verdicts" in text and "H2 not run" in text
    first = (m["paths"].REPORTS / "itemtriage_h1.md").read_text()
    ev.run()
    assert (m["paths"].REPORTS / "itemtriage_h1.md").read_text() == first, "H1 report must be deterministic"
