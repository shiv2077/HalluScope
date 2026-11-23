"""Cost logging. Every OpenAI call goes through log_call and lands in results/calls.jsonl.

Prices come only from config/pricing.json (with its priced_on date). No dollar figure is typed anywhere else.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .data import CONFIG, RESULTS

CALLS = RESULTS / "calls.jsonl"
_PRICING = json.loads((CONFIG / "pricing.json").read_text())


def priced_on() -> str:
    return _PRICING["priced_on"]


def price_source(model: str) -> str:
    return _PRICING["models"][model]["source"]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _PRICING["models"].get(model)
    if p is None:
        raise KeyError(f"{model} has no entry in config/pricing.json")
    return (input_tokens * p["input_per_m"] + output_tokens * p["output_per_m"]) / 1e6


def require_keys(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"Missing API key(s): {', '.join(missing)}. Put them in .env. "
                         "This build never simulates or mocks data.")


@contextmanager
def log_call(model: str, purpose: str):
    """Yield a record dict; the caller fills input_tokens and output_tokens. Appended on exit."""
    rec = {"model": model, "purpose": purpose, "input_tokens": None, "output_tokens": None,
           "t_start": time.time()}
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.time() - rec.pop("t_start"), 3)
        if rec["input_tokens"] is not None:
            rec["cost_usd"] = cost_usd(model, rec["input_tokens"], rec["output_tokens"])
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(CALLS, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def summary(path: Path = CALLS) -> dict:
    """Total calls, tokens and cost from calls.jsonl, overall and per model and purpose."""
    out = {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "by_model": {},
           "by_purpose": {}, "priced_on": priced_on()}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("input_tokens") is None:
            continue
        for key, bucket in (("by_model", r["model"]), ("by_purpose", r["purpose"])):
            b = out[key].setdefault(bucket, {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            b["n_calls"] += 1
            b["input_tokens"] += r["input_tokens"]
            b["output_tokens"] += r["output_tokens"]
            b["cost_usd"] += r["cost_usd"]
        out["n_calls"] += 1
        out["input_tokens"] += r["input_tokens"]
        out["output_tokens"] += r["output_tokens"]
        out["cost_usd"] += r["cost_usd"]
    return out
