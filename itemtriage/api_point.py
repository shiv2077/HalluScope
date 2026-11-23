"""Study 1 with its preregistered API models, run after unblinding (PREREG Amendment 1). Exploratory.

The OpenAI key was invalid when Study 1 ran, so the panel was five local models. With a valid key, gpt-4o and
gpt-4o-mini ran on every item on 24 November 2026, after the test fold was unblinded. This module scores the
preregistered seven-model panel with Study 1's own scorers (including ptrue_key) and the two-model API panel,
and evaluates both on the Study 1 test set. It writes results/itemtriage/api_point.json and never touches
Study 1's files. Requires `itemtriage.cli collect` after the API pass.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import collect, costs, evaluate, panel_api, panel_local, paths, rank, scorers

OUT = paths.RESULTS / "api_point.json"
PANEL7 = list(panel_api.API_MODELS) + list(panel_local.PANEL_LOCAL)
NAMES7 = ("conf_w", "conf", "disagree", "irt_small", "ptrue_key", "length")
PAIRS7 = [("conf_w", "disagree"), ("conf_w", "irt_small"), ("ptrue_key", "conf_w")]
LABEL = "Exploratory, post-unblinding: the API models ran on 24 November 2026, after the test fold was unblinded."


def api_scores() -> pd.DataFrame:
    """disagree and conf for the two-model API panel, plus ptrue_key. Label-free."""
    pr = pd.read_parquet(collect.PANEL_OUT)
    pr = pr[pr["model"].isin(panel_api.API_MODELS)]
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")
    pk = pr.pivot(index="item_id", columns="model", values="p_key")
    am = pr.pivot(index="item_id", columns="model", values="argmax_index")
    key = items["key_index"].reindex(pk.index).to_numpy()
    out = pd.DataFrame({"item_id": pk.index})
    out["disagree"] = (am.to_numpy() != key[:, None]).mean(axis=1)
    out["conf"] = (1.0 - pk.to_numpy()).mean(axis=1)
    pt = pd.read_parquet(collect.PTRUE_OUT)
    pt = pt[pt["model"].isin(panel_api.API_MODELS)].pivot(index="item_id", columns="model", values="logodds_no_yes")
    out["ptrue_key"] = pt.reindex(out["item_id"]).mean(axis=1).to_numpy()
    return out


def usd_per_1000_items() -> dict:
    """Mean logged cost per item and model, by purpose, times 1,000. From results/itemtriage/calls.jsonl."""
    rows = [json.loads(x) for x in costs.CALLS.read_text().splitlines() if x.strip()]
    out = {}
    for m in panel_api.API_MODELS:
        for pu in ("mc", "ptrue"):
            c = [r["cost_usd"] for r in rows if r["model"] == m and r["purpose"].startswith(pu) and r.get("cost_usd") is not None]
            out[f"{m}:{pu}"] = {"calls": len(c), "usd_per_1000_items": 1000 * float(np.mean(c)) if c else None}
    return out


def run() -> dict:
    f = evaluate.frame(caller="itemtriage.api_point")
    test = f[(f["fold"] == "test") & f["flaw_primary"].notna()]
    ids = test["item_id"].to_numpy()
    y = test["flaw_primary"].to_numpy(float)
    idx = evaluate.boot_indices(len(test))

    s7, info7 = scorers.build(PANEL7)
    s7 = s7.set_index("item_id").loc[ids]
    S7 = {k: (s7[k].to_numpy(float), s7[scorers.TIE_KEY[k]].to_numpy(float)) for k in NAMES7}
    panel7 = evaluate.evaluate_scorers(y, S7, idx, PAIRS7)

    a = api_scores().set_index("item_id").loc[ids]
    tie = {"disagree": "conf", "conf": "disagree", "ptrue_key": "conf"}
    S2 = {k: (a[k].to_numpy(float), a[tie[k]].to_numpy(float)) for k in tie}
    panel2 = evaluate.evaluate_scorers(y, S2, idx, [("conf", "disagree")])

    cost = usd_per_1000_items()
    mc = sum(v["usd_per_1000_items"] or 0.0 for k, v in cost.items() if k.endswith(":mc"))
    k10 = rank.budget_k(len(y), evaluate.PRIMARY_BUDGET)
    out = {"label": LABEL, "items": len(y), "positives": int(y.sum()), "flagged_at_10": k10,
           "panel7": {"models": PANEL7, "theta": info7["theta"], "result": panel7},
           "panel2": {"models": list(panel_api.API_MODELS), "result": panel2},
           "cost": {"per_model_purpose": cost, "api_panel_mc_usd_per_1000_items": mc}}
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out
