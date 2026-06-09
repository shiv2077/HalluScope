"""Verification for Task 2.5 — local mode end-to-end through FastAPI.

Uses TestClient so we don't need to spin up a real uvicorn process.
Runs one POST /chat with evaluation_mode='local' and asserts the
response carries mean_semantic_energy, mean_rcd, min_rcd, and
hri_mode='local_internal'.

Then GETs /metrics/{id} and asserts internal_per_token is non-empty.

Generation uses the default 80-token cap; on GPU this is a few seconds.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use a throwaway DB for the verification
DB = Path("halluscope_e2e_test.db")
if DB.exists():
    DB.unlink()
os.environ.setdefault("HALLUSCOPE_DB", str(DB))

# Patch storage.DB_PATH after import so we don't pollute the real DB
from backend.app import storage

storage.DB_PATH = DB

from fastapi.testclient import TestClient
from backend.app.main import app

with TestClient(app) as client:
    # Wait for the local model warmup thread (FIX 5) to finish, since
    # we're about to use it. Poll /health.
    print("Waiting for local model warmup...")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        h = client.get("/health").json()
        if h.get("local_model_loaded"):
            print(f"  loaded ({h['local_model_name']})")
            break
        time.sleep(1)
    else:
        raise RuntimeError("Local model failed to warm up within 180s")

    print()
    print("POST /chat (local mode, prompt='Tell me about photosynthesis')")
    t0 = time.monotonic()
    r = client.post(
        "/chat",
        json={
            "message": "Tell me about photosynthesis",
            "evaluation_mode": "local",
        },
    )
    elapsed = time.monotonic() - t0
    print(f"  -> {r.status_code} in {elapsed:.1f}s")
    assert r.status_code == 200, r.text

    body = r.json()
    print()
    print("--- ChatResponse fields of interest ---")
    print(f"  evaluation_mode      : {body['evaluation_mode']}")
    print(f"  hri                  : {body['hri']}")
    print(f"  risk_level           : {body['risk_level']}")
    print(f"  metrics.hri_mode     : {body['metrics']['hri_mode']}")
    print(f"  metrics.mean_energy  : {body['metrics']['mean_semantic_energy']}")
    print(f"  metrics.mean_rcd     : {body['metrics']['mean_rcd']}")
    print(f"  metrics.min_rcd      : {body['metrics']['min_rcd']}")
    print(
        f"  internal_per_token   : keys={list((body.get('internal_per_token') or {}).keys())}"
    )

    # ── Assertions ──────────────────────────────────────────────────
    assert body["evaluation_mode"] == "local"
    assert (
        body["metrics"]["hri_mode"] == "local_internal"
    ), f"hri_mode must be 'local_internal' for local mode, got {body['metrics']['hri_mode']}"
    assert body["metrics"]["mean_semantic_energy"] is not None
    assert body["metrics"]["mean_rcd"] is not None
    assert body["metrics"]["min_rcd"] is not None
    ipt = body.get("internal_per_token")
    assert ipt is not None, "internal_per_token missing"
    assert len(ipt["energy"]) > 0, "internal_per_token.energy empty"
    # RCD has length n-1 (no value for t=0)
    assert len(ipt["rcd"]) == max(0, len(ipt["energy"]) - 1)
    print()
    print("OK: all local-mode response assertions pass")

    # /metrics round-trip
    mid = body["message_id"]
    print(f"\nGET /metrics/{mid}")
    rm = client.get(f"/metrics/{mid}")
    assert rm.status_code == 200, rm.text
    mb = rm.json()
    assert mb["metrics"]["hri_mode"] == "local_internal"
    assert mb["internal_per_token"] is not None
    assert mb["internal_per_token"]["energy"] == ipt["energy"]
    print("OK: /metrics round-trip preserves internal_per_token + hri_mode")

    # ── Sanity: API mode must still produce hri_mode='api' and no internals
    # We don't hit OpenAI in this test (no key, would cost money) — just
    # verify the schema path is intact by checking that compute_hri called
    # with no internal_signals returns hri_mode='api'. Already covered in
    # the Task 2.4 test, so just print a reminder.
    print()
    print(
        "(API mode hri_mode='api' path covered by test_internal_metrics + Task 2.4 inline test)"
    )

if DB.exists():
    DB.unlink()

print()
print("ALL TASK 2.5 ASSERTIONS PASSED.")
