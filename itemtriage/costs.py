"""Spend accounting and the cost gate. Prices and summaries come from routing.cost; calls log to
results/itemtriage/calls.jsonl.

The $25 cap covers the whole repository: routing's results/calls.jsonl plus itemtriage's log.
Before a pass, project its cost from token counts (rendered characters / 4 before a pilot, measured mean
tokens per call after one) and refuse if projection plus spend so far would exceed the cap.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

from routing import cost as rcost

from . import paths

CAP_USD = 25.0
CHAT_OVERHEAD_TOKENS = 7  # per-message framing tokens added by the chat format
CALLS = paths.RESULTS / "calls.jsonl"
cost_usd = rcost.cost_usd


@contextmanager
def log_call(model: str, purpose: str):
    """Same record format as routing.cost.log_call, written to the itemtriage call log."""
    rec = {"model": model, "purpose": purpose, "input_tokens": None, "output_tokens": None, "t_start": time.time()}
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.time() - rec.pop("t_start"), 3)
        if rec["input_tokens"] is not None:
            rec["cost_usd"] = cost_usd(model, rec["input_tokens"], rec["output_tokens"])
        CALLS.parent.mkdir(parents=True, exist_ok=True)
        with open(CALLS, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def spent() -> dict:
    r = rcost.summary(paths.ROUTING_CALLS)
    i = rcost.summary(CALLS)
    return {"routing_usd": r["cost_usd"], "itemtriage_usd": i["cost_usd"], "total_usd": r["cost_usd"] + i["cost_usd"],
            "itemtriage_calls": i["n_calls"], "priced_on": rcost.priced_on()}


def measured_tokens(model: str, purpose_prefix: str) -> tuple[float, float] | None:
    """Mean input and output tokens per call for this model and purpose, from the itemtriage log."""
    if not CALLS.exists():
        return None
    rows = [json.loads(x) for x in CALLS.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["model"] == model and r["purpose"].startswith(purpose_prefix) and r.get("input_tokens") is not None]
    if not rows:
        return None
    return (sum(r["input_tokens"] for r in rows) / len(rows), sum(r["output_tokens"] for r in rows) / len(rows))


def project(plan: list[dict]) -> dict:
    """plan rows: {model, purpose, n_calls, est_input_tokens, est_output_tokens}. Measured means win when present."""
    lines, total = [], 0.0
    for p in plan:
        m = measured_tokens(p["model"], p["purpose"])
        ti, to = m if m else (p["est_input_tokens"], p["est_output_tokens"])
        usd = p["n_calls"] * cost_usd(p["model"], ti, to) if p["n_calls"] else 0.0
        lines.append({**p, "input_tokens_per_call": round(ti, 1), "output_tokens_per_call": round(to, 2),
                      "basis": "measured" if m else "estimate", "usd": usd})
        total += usd
    s = spent()
    return {"lines": lines, "projected_usd": total, **s, "after_usd": s["total_usd"] + total, "cap_usd": CAP_USD,
            "within_cap": s["total_usd"] + total <= CAP_USD}


def gate(plan: list[dict]) -> dict:
    pr = project(plan)
    if not pr["within_cap"]:
        raise SystemExit(f"cost gate: projected ${pr['projected_usd']:.2f} plus spent ${pr['total_usd']:.2f} "
                         f"exceeds the ${CAP_USD:.0f} cap")
    return pr


def render(pr: dict) -> str:
    L = [f"Cost projection (prices from config/pricing.json, priced_on {pr['priced_on']}):",
         f"{'model':<12} {'purpose':<22} {'calls':>7} {'in/call':>8} {'out/call':>8} {'basis':<9} {'usd':>8}"]
    for x in pr["lines"]:
        L.append(f"{x['model']:<12} {x['purpose']:<22} {x['n_calls']:>7} {x['input_tokens_per_call']:>8} "
                 f"{x['output_tokens_per_call']:>8} {x['basis']:<9} {x['usd']:>8.3f}")
    L.append(f"projected ${pr['projected_usd']:.2f}; spent so far ${pr['total_usd']:.2f} "
             f"(routing ${pr['routing_usd']:.2f}, itemtriage ${pr['itemtriage_usd']:.2f}); "
             f"after ${pr['after_usd']:.2f} of ${pr['cap_usd']:.0f}; within cap: {pr['within_cap']}")
    return "\n".join(L)
