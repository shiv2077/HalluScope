"""Study 2 panels and label-free scorers (docs/PREREG_2.md). Never reads labels.

Panels are drawn from the 335 representative leaderboard models by strength and size. Scorers run on the
binary correctness matrix (and on P(key) where fetched). IRT fits are slow, so their outputs are cached in
results/itemtriage/study2_irt.parquet and study2_theta.json and refit only on request.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import irt, matrix, paths

SEED = 42
SIZES = (1, 3, 5, 7, 10, 20, 50)
DRAWS = 50
IRT_DRAWS = 20
IRT_SIZES = (5, 10, 20, 50)
MIN_K = 5           # item_total, irt_small and conf_w need at least this many models
STRONG = 3          # strong3 uses the panel's 3 most able models
CHANCE = 0.25
IRT_CACHE = paths.RESULTS / "study2_irt.parquet"
THETA_CACHE = paths.RESULTS / "study2_theta.json"


# ------------------------------------------------------------------ panels
def representatives() -> pd.DataFrame:
    """Representative models, most able first, ties by name."""
    m = pd.read_parquet(matrix.MODELS).query("representative")
    return m.sort_values(["ability", "model"], ascending=[False, True]).reset_index(drop=True)


def draw_panels(reps: pd.DataFrame, sizes: tuple[int, ...] = SIZES) -> list[dict]:
    """Every panel in the preregistered grid, in a fixed order. Each: id, stratum, k, draw, models (most able first)."""
    names = reps["model"].tolist()
    n = len(names)
    out = []

    def add(stratum, k, d, idx):
        idx = sorted(int(i) for i in idx)
        out.append({"id": f"{stratum}-{k}-{d}", "stratum": stratum, "k": k, "draw": d, "models": [names[i] for i in idx]})

    for k in sizes:
        add("top", k, 0, range(k))
        add("bottom", k, 0, range(n - k, n))
    rng = np.random.RandomState(SEED)
    for k in sizes:
        for d in range(DRAWS):
            add("random", k, d, rng.choice(n, k, replace=False))
    lin = reps["lineage"].to_numpy()
    groups = {g: np.flatnonzero(lin == g) for g in np.unique(lin)}
    keys = np.array(sorted(groups))
    rng = np.random.RandomState(SEED)
    for k in sizes:
        for d in range(DRAWS):
            chosen = rng.choice(keys, k, replace=False)
            add("lineage", k, d, [rng.choice(groups[g]) for g in chosen])
    add("all", n, 0, range(n))
    return out


def needs_irt(p: dict) -> bool:
    if p["k"] < MIN_K:
        return False
    if p["stratum"] in ("top", "bottom", "all"):
        return True
    return p["k"] in IRT_SIZES and p["draw"] < IRT_DRAWS


# ------------------------------------------------------------------ scorers
def matrix_inputs() -> dict:
    """Correctness and P(key) aligned to MMLU-Redux items, with subjects and length. Label-free."""
    corr = pd.read_parquet(matrix.CORRECT)
    items = pd.read_parquet(paths.ITEMS).set_index("item_id").loc[corr["item_id"]]
    pk = pd.read_parquet(matrix.PKEY) if matrix.PKEY.exists() else None
    length = np.array([len(q) + sum(len(o) for o in opts) for q, opts in zip(items["question"], items["options"])], float)
    return {"item_id": corr["item_id"].to_numpy(), "subject": items["subject"].to_numpy(), "Y": corr.drop(columns="item_id"),
            "P": pk.set_index("item_id").loc[corr["item_id"]] if pk is not None else None, "length": length}


def item_total(Y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Minus the within-group item-total correlation. 0 where the response vector is constant."""
    return -np.nan_to_num(irt.item_total_r(Y, groups), nan=0.0)


def cheap_scores(p: dict, inp: dict, theta: list | None) -> dict[str, np.ndarray]:
    """Every scorer except irt_small. theta: stage-1 IRT ability of the panel members, or None if not fit."""
    Y = inp["Y"][p["models"]].to_numpy(float)
    s = {"disagree": 1.0 - Y.mean(axis=1), "strong3": 1.0 - Y[:, :STRONG].mean(axis=1)}
    s["subject_mean"] = pd.Series(s["disagree"]).groupby(inp["subject"]).transform("mean").to_numpy()
    if p["k"] >= MIN_K:
        s["item_total"] = item_total(Y, inp["subject"])
    P = inp["P"]
    if P is not None:
        have = [m for m in p["models"] if m in P.columns]
        if have:
            miss = 1.0 - P[have].to_numpy(float)
            s["conf"] = miss.mean(axis=1)
            if theta is not None:
                th = np.array([theta[p["models"].index(m)] for m in have])
                w = np.exp(th - th.max())
                s["conf_w"] = miss @ (w / w.sum())
    return s


def fit_irt(args) -> tuple[str, list, np.ndarray]:
    """Stage-1 ability over all items, then Δℓ on the evaluation items only. Runs in a worker process."""
    pid, models, eval_idx = args
    inp = _WORKER["inp"]
    Y = inp["Y"][models].to_numpy(float)
    s1 = irt.fit_stage1(Y, inp["subject"], CHANCE)
    return pid, s1["theta"].tolist(), irt.delta_full(Y[eval_idx], s1["theta"], CHANCE)


_WORKER: dict = {}


def _init(inp):
    _WORKER["inp"] = inp


def refit_irt(panels: list[dict], eval_ids: np.ndarray, workers: int = 8) -> None:
    """Fit IRT for every panel that needs it and cache the results. Resumes from a partial cache."""
    from multiprocessing import get_context

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    inp = matrix_inputs()
    pos = {i: n for n, i in enumerate(inp["item_id"])}
    eval_idx = np.array([pos[i] for i in eval_ids])
    theta = json.loads(THETA_CACHE.read_text()) if THETA_CACHE.exists() else {}
    have = pd.read_parquet(IRT_CACHE) if IRT_CACHE.exists() else pd.DataFrame(columns=["panel", "item_id", "irt_small"])
    todo = [(p["id"], p["models"], eval_idx) for p in panels if needs_irt(p) and p["id"] not in theta]
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(workers, initializer=_init, initargs=(inp,)) as pool:
        for n, (pid, th, d) in enumerate(pool.imap_unordered(fit_irt, todo), 1):
            theta[pid] = th
            have = pd.concat([have, pd.DataFrame({"panel": pid, "item_id": eval_ids, "irt_small": d})], ignore_index=True)
            if n % 8 == 0 or n == len(todo):
                have.sort_values(["panel", "item_id"]).to_parquet(IRT_CACHE, index=False)
                THETA_CACHE.write_text(json.dumps(theta, sort_keys=True))
            print(f"irt {n}/{len(todo)}", flush=True)
    have.sort_values(["panel", "item_id"]).reset_index(drop=True).to_parquet(IRT_CACHE, index=False)
    THETA_CACHE.write_text(json.dumps(theta, sort_keys=True))


def irt_scores() -> tuple[dict, dict]:
    """Cached Δℓ per panel (dict panel -> Series by item_id) and stage-1 theta per panel."""
    if not IRT_CACHE.exists():
        return {}, {}
    d = pd.read_parquet(IRT_CACHE)
    return {pid: g.set_index("item_id")["irt_small"] for pid, g in d.groupby("panel")}, json.loads(THETA_CACHE.read_text())
