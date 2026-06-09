"""E0 harness smoke as a pytest (tiny: 2 questions on 0.5B, CPU-ok).

Marked `slow` because it loads a real model. Run just this with:
    .venv/bin/python -m pytest tests/hv2/test_harness_smoke.py -q -m slow
The fuller 5-question check lives in scripts/smoke_e0.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.mark.slow
def test_e0_smoke_minimal(tmp_path):
    from scripts.smoke_e0 import run

    summary = run(n=2, cache_path=str(tmp_path / "smoke.h5"))
    assert summary["n_records"] == 2
    assert summary["device"] in ("cpu", "cuda")
    assert summary["peak_rss_mb"] > 0
