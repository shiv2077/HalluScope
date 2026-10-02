"""Item scorers from the panel's responses. Label-free; higher is riskier. See docs/PREREG.md.

disagree, conf, conf_w (primary), irt_small, irt_small_c, ptrue_key, and the controls length and
dup_options. IRT is fit transductively on the full response matrix (correctness against the key).
Writes results/itemtriage/scores.parquet and results/itemtriage/irt.json.
"""

from __future__ import annotations

import difflib
import json
from itertools import combinations

import numpy as np
import pandas as pd

from . import collect, irt, paths
from .ingest import normalize

SCORES = paths.RESULTS / "scores.parquet"
IRT_OUT = paths.RESULTS / "irt.json"
PANEL_FILE = paths.RESULTS / "panel.json"
CHANCE = 0.25
TIE_KEY = {"disagree": "conf", "conf": "disagree", "conf_w": "conf", "irt_small": "conf", "irt_small_c": "conf",
           "ptrue_key": "conf", "length": "conf", "dup_options": "conf"}
LABEL_FREE = tuple(TIE_KEY)


def panel_models() -> list[str]:
    return json.loads(PANEL_FILE.read_text())["panel"]


def dup_options(options: list) -> float:
    o = [normalize(x) for x in options]
    return max(difflib.SequenceMatcher(None, a, b).ratio() for a, b in combinations(o, 2))


def build(models: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    models = models or panel_models()
    items = pd.read_parquet(paths.ITEMS).sort_values("item_id").reset_index(drop=True)
    pr = pd.read_parquet(collect.PANEL_OUT)
    pr = pr[pr["model"].isin(models)]
    pk = pr.pivot(index="item_id", columns="model", values="p_key").reindex(items["item_id"])[models]
    am = pr.pivot(index="item_id", columns="model", values="argmax_index").reindex(items["item_id"])[models]
    if pk.isna().any().any():
        missing = int(pk.isna().any(axis=1).sum())
        raise ValueError(f"{missing} items lack a response from some panel model; finish the panel pass first")
    key = items.set_index("item_id")["key_index"].reindex(pk.index).to_numpy()
    Y = (am.to_numpy() == key[:, None]).astype(float)
    groups = items["subject"].to_numpy()
    s1 = irt.fit_stage1(Y, groups, CHANCE)
    theta = s1["theta"]
    w = np.exp(theta) / np.exp(theta).sum()
    miss = 1.0 - pk.to_numpy()
    out = pd.DataFrame({"item_id": items["item_id"], "subject": items["subject"]})
    out["disagree"] = (1.0 - Y).mean(axis=1)
    out["conf"] = miss.mean(axis=1)
    out["conf_w"] = miss @ w
    out["irt_small"] = irt.delta_full(Y, theta, CHANCE)
    out["irt_small_c"] = irt.delta_constrained(Y, theta, groups, s1["a_group"], CHANCE)
    pt = pd.read_parquet(collect.PTRUE_OUT)
    api = [m for m in models if m in set(pt["model"])]
    if api:
        out["ptrue_key"] = pt[pt["model"].isin(api)].pivot(index="item_id", columns="model",
                                                             values="logodds_no_yes").reindex(items["item_id"]).mean(axis=1).to_numpy()
    else:
        out["ptrue_key"] = np.nan
    out["length"] = [len(q) + sum(len(o) for o in opts) for q, opts in zip(items["question"], items["options"])]
    out["dup_options"] = [dup_options(list(o)) for o in items["options"]]
    info = {"models": models, "theta": dict(zip(models, theta.tolist())), "weights": dict(zip(models, w.tolist())),
            "a_group": s1["a_group"], "stage1_items_kept": s1["n_kept"], "stage1_converged": s1["converged"],
            "stage1_message": s1["message"], "ptrue_models": api, "tie_keys": TIE_KEY}
    return out, info


def run(models: list[str] | None = None) -> dict:
    out, info = build(models)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SCORES, index=False)
    IRT_OUT.write_text(json.dumps(info, indent=1, sort_keys=True))
    return {"items": len(out), "models": info["models"], "stage1_items_kept": info["stage1_items_kept"],
            "converged": info["stage1_converged"]}
