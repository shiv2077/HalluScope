"""API panel inference: one direct-letter call and one P(True) call per item and model. Label-free.

Raw responses are appended to data/itemtriage/raw/api_<model>_<purpose>.jsonl as each arrives, keyed by
item_id, so a pass resumes from where it stopped. Every call is logged via itemtriage.costs.log_call.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import costs, paths, prompts

API_MODELS = ("gpt-4o", "gpt-4o-mini")
PURPOSES = ("mc", "ptrue")
TOP_LOGPROBS = 20
SEED = 42
WORKERS = 8
PILOT_N, PILOT_SEED = 300, 44
_lock = threading.Lock()


def raw_path(model: str, purpose: str):
    return paths.RAW / f"api_{model}_{purpose}.jsonl"


def done_ids(model: str, purpose: str) -> set:
    p = raw_path(model, purpose)
    if not p.exists():
        return set()
    return {json.loads(x)["item_id"] for x in p.read_text().splitlines() if x.strip()}


def pilot_ids() -> list[str]:
    from . import splits

    fit = sorted(splits.load()["item_ids"]["calibration_fit"])
    return sorted(np.random.RandomState(PILOT_SEED).choice(fit, PILOT_N, replace=False).tolist())


def render(item: dict, purpose: str) -> str:
    if purpose == "mc":
        text, order = prompts.mc_prompt(item, 0)
        prompts.check_prompt(text, item, order)
    else:
        text = prompts.ptrue_prompt(item)
        prompts.check_prompt(text, item, None)
    return text


def _call(client, model: str, purpose: str, tag: str, item: dict) -> None:
    text = render(item, purpose)
    with costs.log_call(model, f"{purpose}:{tag}") as rec:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}], temperature=0,
                                           max_tokens=1, logprobs=True, top_logprobs=TOP_LOGPROBS, seed=SEED)
        rec["input_tokens"], rec["output_tokens"] = r.usage.prompt_tokens, r.usage.completion_tokens
    ch = r.choices[0]
    top = [{"token": a.token, "logprob": a.logprob} for a in ch.logprobs.content[0].top_logprobs] if ch.logprobs and ch.logprobs.content else []
    row = {"item_id": item["item_id"], "model": model, "purpose": purpose, "tag": tag, "text": ch.message.content or "",
           "top_logprobs": top, "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens,
           "system_fingerprint": r.system_fingerprint, "response_model": r.model}
    with _lock:
        paths.RAW.mkdir(parents=True, exist_ok=True)
        with open(raw_path(model, purpose), "a") as fh:
            fh.write(json.dumps(row) + "\n")


def plan_for(ids: list[str], tag: str) -> list[dict]:
    items = pd.read_parquet(paths.ITEMS).set_index("item_id").loc[ids].reset_index().to_dict("records")
    est = {pu: sum(len(render(i, pu)) for i in items) / max(len(items), 1) / 4 + costs.CHAT_OVERHEAD_TOKENS for pu in PURPOSES}
    out = []
    for m in API_MODELS:
        for pu in PURPOSES:
            todo = len(set(ids) - done_ids(m, pu))
            out.append({"model": m, "purpose": f"{pu}:{tag}", "n_calls": todo, "est_input_tokens": est[pu], "est_output_tokens": 1})
    return out


def run(ids: list[str], tag: str) -> dict:
    costs.rcost.require_keys("OPENAI_API_KEY")
    pr = costs.gate(plan_for(ids, tag))
    print(costs.render(pr), file=sys.stderr)
    import openai

    client = openai.OpenAI()
    try:
        available = {m.id for m in client.models.list().data}
    except openai.AuthenticationError as e:
        raise SystemExit(f"OpenAI rejected the key before any call: {e}. Put a valid OPENAI_API_KEY in .env.")
    missing = set(API_MODELS) - available
    if missing:
        raise SystemExit(f"models not available to this key: {sorted(missing)}")
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")
    jobs = [(m, pu, items.loc[i].to_dict() | {"item_id": i}) for m in API_MODELS for pu in PURPOSES
            for i in sorted(set(ids) - done_ids(m, pu))]
    errors = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_call, client, m, pu, tag, it) for m, pu, it in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except openai.AuthenticationError as e:
                for g in futs:
                    g.cancel()
                raise SystemExit(f"authentication failed mid-pass, stopping: {e}")
            except Exception as e:  # recorded and retried on the next run, never silently dropped
                errors += 1
                print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            if k % 500 == 0:
                print(f"{tag}: {k}/{len(jobs)}", file=sys.stderr)
    return {"jobs": len(jobs), "errors": errors, "spent": costs.spent()}
