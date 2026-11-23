"""Panel selection on the 300-item calibration pilot (docs/PREREG.md). Label-free: accuracy is against the key.

A model stays if its letter parse rate is at least 95% and the Wilson lower bound of its accuracy exceeds
chance (0.25). A failing local model is replaced by the reserve. Fewer than 5 passing models stops the study.
Writes results/itemtriage/pilot.json and results/itemtriage/panel.json.
"""

from __future__ import annotations

import json

import pandas as pd

from routing import stats

from . import collect, panel_api, panel_local, paths

PARSE_MIN, CHANCE, MIN_PANEL = 0.95, 0.25, 5
RESERVE = "olmo-2-1b"


def evaluate(models: list[str]) -> dict:
    ids = set(panel_api.pilot_ids())
    pr = pd.read_parquet(collect.PANEL_OUT)
    pr = pr[pr["item_id"].isin(ids)]
    key = pd.read_parquet(paths.ITEMS).set_index("item_id")["key_index"]
    out = {}
    for m in models:
        g = pr[pr["model"] == m]
        n = len(g)
        if n == 0:
            reason = "not run: the OpenAI key was invalid (PREREG Amendment 1)" if m in panel_api.API_MODELS else "no pilot responses"
            out[m] = {"n": 0, "passes": False, "reason": reason}
            continue
        acc = stats.wilson(int((g["argmax_index"].to_numpy() == key.loc[g["item_id"]].to_numpy()).sum()), n)
        parse = stats.wilson(int(g["parse_ok"].sum()), n)
        agree = stats.wilson(int(g["orders_agree"].sum()), n)
        ok = n == len(ids) and parse["est"] >= PARSE_MIN and acc["lo"] > CHANCE
        out[m] = {"n": n, "accuracy": acc, "parse_rate": parse, "orders_agree": agree, "passes": bool(ok)}
    return out


def select() -> dict:
    candidates = list(panel_api.API_MODELS) + list(panel_local.PANEL_LOCAL)
    res = evaluate(candidates + [RESERVE])
    panel = [m for m in candidates if res[m]["passes"]]
    failed_local = [m for m in panel_local.PANEL_LOCAL if not res[m]["passes"]]
    if failed_local and res[RESERVE]["passes"]:
        panel.append(RESERVE)
    body = {"pilot_items": len(panel_api.pilot_ids()), "results": res, "panel": panel, "replaced": failed_local,
            "stop": len(panel) < MIN_PANEL}
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "pilot.json").write_text(json.dumps(body, indent=1, sort_keys=True, default=float))
    if not body["stop"]:
        (paths.RESULTS / "panel.json").write_text(json.dumps({"panel": panel}, indent=1))
    return body
