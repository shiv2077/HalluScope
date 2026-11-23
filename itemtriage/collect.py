"""Turn raw panel responses into one row per (item, model). Label-free and deterministic.

API: letter distribution from the first-token top 20; P(True) log-odds from the second call.
Local: per-order letter logprobs renormalized over letters, mapped back to original options, and averaged
over the two orders in probability space. The first raw row per key wins if a pass was repeated.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from . import panel_api, panel_local, parse, paths

PANEL_OUT = paths.DATA / "panel_responses.parquet"
PTRUE_OUT = paths.DATA / "ptrue_responses.parquet"


def _rows(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _first(rows, key):
    seen, out = set(), []
    for r in rows:
        k = key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def collect(api_models=panel_api.API_MODELS, local_slugs=panel_local.PANEL_LOCAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")
    nopt = items["n_options"].to_dict()
    key = items["key_index"].to_dict()
    panel, ptrue = [], []
    for m in api_models:
        for r in _first(_rows(panel_api.raw_path(m, "mc")), lambda r: r["item_id"]):
            if not r["top_logprobs"]:
                continue
            lp, missing, ok = parse.letter_logprobs(r["top_logprobs"], nopt[r["item_id"]])
            p = [math.exp(v) for v in lp]
            panel.append({"item_id": r["item_id"], "model": m, "family": "OpenAI", "access": "api", "p_options": p,
                          "p_key": p[key[r["item_id"]]], "argmax_index": max(range(len(p)), key=lambda j: (p[j], -j)),
                          "parse_ok": ok, "letters_missing": missing, "n_orders": 1, "orders_agree": True})
        for r in _first(_rows(panel_api.raw_path(m, "ptrue")), lambda r: r["item_id"]):
            if not r["top_logprobs"]:
                continue
            lo, ya, na = parse.p_true_logodds(r["top_logprobs"])
            ptrue.append({"item_id": r["item_id"], "model": m, "logodds_no_yes": lo, "yes_absent": ya, "no_absent": na})
    for slug in local_slugs:
        name, _q, fam = panel_local.MODELS[slug]
        by_item: dict = {}
        for r in _first(_rows(panel_local.raw_path(slug)), lambda r: (r["item_id"], r["shift"])):
            by_item.setdefault(r["item_id"], []).append(r)
        for iid, rs in sorted(by_item.items()):
            probs, argmaxes = [], []
            for r in sorted(rs, key=lambda r: r["shift"]):
                z = max(r["letter_logprobs"])
                e = [math.exp(v - z) for v in r["letter_logprobs"]]
                q = parse.unpermute([v / sum(e) for v in e], r["order"])
                probs.append(q)
                argmaxes.append(max(range(len(q)), key=lambda j: (q[j], -j)))
            p = [sum(c) / len(probs) for c in zip(*probs)]
            panel.append({"item_id": iid, "model": slug, "family": fam, "access": "local", "p_options": p,
                          "p_key": p[key[iid]], "argmax_index": max(range(len(p)), key=lambda j: (p[j], -j)),
                          "parse_ok": all(r["top1_is_letter"] for r in rs), "letters_missing": 0, "n_orders": len(rs),
                          "orders_agree": len(set(argmaxes)) == 1})
    pdf = pd.DataFrame(panel).sort_values(["item_id", "model"]).reset_index(drop=True)
    tdf = pd.DataFrame(ptrue).sort_values(["item_id", "model"]).reset_index(drop=True) if ptrue else pd.DataFrame(
        columns=["item_id", "model", "logodds_no_yes", "yes_absent", "no_absent"])
    return pdf, tdf


def run(**kw) -> dict:
    pdf, tdf = collect(**kw)
    pdf.to_parquet(PANEL_OUT, index=False)
    tdf.to_parquet(PTRUE_OUT, index=False)
    return {"panel_rows": len(pdf), "ptrue_rows": len(tdf), "models": sorted(pdf["model"].unique()) if len(pdf) else []}
