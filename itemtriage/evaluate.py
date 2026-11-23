"""H1 evaluation on the test fold, under the preregistration. Labels are read here and nowhere upstream.

Refuses to run unless docs/PREREG.md is tracked and unmodified. The first run writes
results/itemtriage/unblinding.json; a later run under a different PREREG commit is flagged as a deviation.
Bootstrap: 2000 resamples, seed 42, the same index draws as routing.stats.boot, shared across all statistics
so paired differences use identical resamples. Budget metrics resolve ties in expectation (itemtriage.rank).
Writes results/itemtriage/evaluation.json and reports/itemtriage_h1.md.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from routing import stats

from . import costs, paths, rank, scorers, splits, supervised
from . import labels as labels_mod

BUDGETS = (0.05, 0.10, 0.20)
PRIMARY_BUDGET = 0.10
CLASSES = {"primary": "flaw_primary", "secondary": "flaw_any"}
PREREG = "docs/PREREG.md"
REPO = paths.ROOT
UNBLIND = paths.RESULTS / "unblinding.json"
EVAL_OUT = paths.RESULTS / "evaluation.json"
N_BOOT, SEED = stats.N_BOOT, stats.SEED
REVIEW_MINUTES_ASSUMED = 2.0  # an assumption, labelled wherever it is used
API_SCORER_PURPOSES = {"disagree": ("mc",), "conf": ("mc",), "conf_w": ("mc",), "irt_small": ("mc",), "irt_small_c": ("mc",),
                       "ptrue_key": ("ptrue",), "lr": ("mc", "ptrue"), "length": (), "dup_options": ()}


# ------------------------------------------------------------------ preregistration guard
def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def prereg_commit() -> str:
    if _git("ls-files", "--error-unmatch", PREREG).returncode:
        raise SystemExit(f"{PREREG} is not tracked by git; commit it before any test evaluation")
    if _git("diff", "--quiet", "HEAD", "--", PREREG).returncode:
        raise SystemExit(f"{PREREG} has uncommitted changes; commit or revert them before any test evaluation")
    return _git("log", "-1", "--format=%H", "--", PREREG).stdout.strip()


def unblind(commit: str) -> dict:
    if not UNBLIND.exists():
        UNBLIND.parent.mkdir(parents=True, exist_ok=True)
        UNBLIND.write_text(json.dumps({"prereg_commit": commit,
                                       "first_test_evaluation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                      indent=1))
    first = json.loads(UNBLIND.read_text())
    return {"prereg_commit": commit, "unblinded_under": first["prereg_commit"],
            "deviation": first["prereg_commit"] != commit}


# ------------------------------------------------------------------ statistics
def boot_indices(n: int) -> list[np.ndarray]:
    rng = np.random.RandomState(SEED)
    return [rng.randint(0, n, n) for _ in range(N_BOOT)]


def _ci(est, vals) -> dict | None:
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if est is None or not np.isfinite(est) or len(vals) == 0:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n_boot_valid": int(len(vals))}


def composite(s1, s2) -> np.ndarray:
    """One score that orders by (s1, s2) and leaves only exact double ties tied, for AUROC."""
    order = np.lexsort((s2, s1))
    a, b = np.asarray(s1)[order], np.asarray(s2)[order]
    new = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])]
    r = np.empty(len(a))
    r[order] = np.cumsum(new)
    return r


def stat_block(y, s1, s2) -> dict:
    out = {}
    base = y.mean()
    for b in BUDGETS:
        p = rank.precision_at(y, s1, s2, b)
        out[f"precision_{b}"] = p
        out[f"recall_{b}"] = rank.recall_at(y, s1, s2, b)
        out[f"lift_{b}"] = p / base if base > 0 else None
    out["ap"] = rank.average_precision(y, s1, s2)
    out["auroc"] = float(roc_auc_score(y, composite(s1, s2))) if 0 < y.sum() < len(y) else None
    return out


def evaluate_scorers(y: np.ndarray, S: dict[str, tuple[np.ndarray, np.ndarray]], idx: list[np.ndarray], pairs: list) -> dict:
    """S: scorer -> (score, tie key). pairs: (a, b) for paired precision differences at the primary budget."""
    est = {k: stat_block(y, *v) for k, v in S.items()}
    reps = {k: {m: [] for m in est[k]} for k in S}
    diffs = {f"{a}-{b}": [] for a, b in pairs}
    for ix in idx:
        yy = y[ix]
        cur = {}
        for k, (s1, s2) in S.items():
            blk = stat_block(yy, s1[ix], s2[ix])
            cur[k] = blk
            for m, v in blk.items():
                reps[k][m].append(v)
        for a, b in pairs:
            diffs[f"{a}-{b}"].append(cur[a][f"precision_{PRIMARY_BUDGET}"] - cur[b][f"precision_{PRIMARY_BUDGET}"])
    res = {k: {m: _ci(est[k][m], reps[k][m]) for m in est[k]} for k in S}
    for k, (s1, s2) in S.items():
        res[k]["ceiling"] = {str(b): min(y.sum(), rank.budget_k(len(y), b)) / rank.budget_k(len(y), b) for b in BUDGETS}
        res[k]["tied_block_at_cutoff"] = {str(b): rank.straddle_size(y, s1, s2, b) for b in BUDGETS}
    paired = {}
    for a, b in pairs:
        e = est[a][f"precision_{PRIMARY_BUDGET}"] - est[b][f"precision_{PRIMARY_BUDGET}"]
        paired[f"{a}-{b}"] = _ci(e, diffs[f"{a}-{b}"])
        if paired[f"{a}-{b}"] is not None:
            paired[f"{a}-{b}"]["share_resamples_at_or_below_zero"] = float(np.mean(np.asarray(diffs[f"{a}-{b}"]) <= 0))
    return {"scorers": res, "paired_precision_diff": paired, "n": int(len(y)), "positives": int(y.sum()),
            "base_rate": stats.wilson(int(y.sum()), len(y))}


# ------------------------------------------------------------------ assembly
def frame(caller: str = "itemtriage.evaluate") -> pd.DataFrame:
    sc = pd.read_parquet(scorers.SCORES)
    lab = labels_mod.load(caller=caller)[["item_id", "flaw_primary", "flaw_any", "error_type_raw"]]
    f = sc.merge(lab, on="item_id", how="left")
    f["fold"] = f["item_id"].map(splits.fold_of(splits.load()))
    assert f["fold"].notna().all()
    return f


def available(f: pd.DataFrame, names: list[str]) -> list[str]:
    """Scorers the panel can produce. ptrue_key needs the API models (PREREG Amendment 1)."""
    return [n for n in names if f[n].notna().all()]


def amendments(models: list[str]) -> list[str]:
    from . import panel_api

    out = []
    if not set(panel_api.API_MODELS) & set(models):
        out.append("Amendment 1 (2 October 2026, before unblinding): the OpenAI key was invalid, so the panel is the five "
                   "local models and ptrue_key is undefined and dropped (docs/PREREG.md).")
    return out


def select_irt(f: pd.DataFrame) -> dict:
    v = f[(f["fold"] == "validation") & f["flaw_primary"].notna()]
    aps = {k: rank.average_precision(v["flaw_primary"].to_numpy(float), v[k].to_numpy(float), v["conf"].to_numpy(float))
           for k in ("irt_small", "irt_small_c")}
    chosen = max(aps, key=lambda k: (aps[k], k == "irt_small"))
    return {"validation_ap": aps, "chosen": chosen}


def api_cost_per_item() -> dict:
    rows = [json.loads(x) for x in costs.CALLS.read_text().splitlines() if x.strip()] if costs.CALLS.exists() else []
    per = {}
    for pu in ("mc", "ptrue"):
        rs = [r for r in rows if r["purpose"].split(":")[0] == pu and r.get("cost_usd") is not None]
        by_model = {}
        for r in rs:
            by_model.setdefault(r["model"], []).append(r["cost_usd"])
        per[pu] = sum(np.mean(v) for v in by_model.values()) if by_model else 0.0
    return per


def memorization(f: pd.DataFrame, folds: tuple[str, ...] = ("calibration_fit", "validation")) -> dict:
    from . import collect

    pr = pd.read_parquet(collect.PANEL_OUT)
    key = pd.read_parquet(paths.ITEMS).set_index("item_id")["key_index"]
    cal = f[f["fold"].isin(list(folds)) & f["flaw_primary"].notna()][["item_id", "flaw_primary"]]
    out = {}
    for m in scorers.panel_models():
        g = pr[pr["model"] == m].merge(cal, on="item_id")
        agree = g["argmax_index"].to_numpy() == key.loc[g["item_id"]].to_numpy()
        fl = g["flaw_primary"].to_numpy() == 1
        out[m] = {"flawed": stats.wilson(int(agree[fl].sum()), int(fl.sum())), "ok": stats.wilson(int(agree[~fl].sum()), int((~fl).sum()))}
    return out


def run() -> dict:
    commit = prereg_commit()
    blind = unblind(commit)
    f = frame()
    info = json.loads(scorers.IRT_OUT.read_text())
    irt_sel = select_irt(f)
    irt_k = irt_sel["chosen"]
    feats = available(f, ["disagree", "conf", "conf_w", irt_k, "ptrue_key", "length", "dup_options"])
    lr = supervised.fit_score(f, feats, "flaw_primary", "conf_w")
    f["lr"] = lr["scores"]
    tie = dict(scorers.TIE_KEY, lr="conf_w")
    names = available(f, ["conf_w", "conf", "disagree", irt_k, "ptrue_key", "lr", "length", "dup_options"])
    test = f[f["fold"] == "test"]
    out = {"prereg": blind, "amendments": amendments(info["models"]), "panel": info["models"], "scorers_evaluated": names,
           "lr_features": feats, "theta": info["theta"], "weights": info["weights"],
           "irt_selection": irt_sel, "lr": {k: v for k, v in lr.items() if k != "scores"},
           "unknown_excluded_test": int(test["flaw_primary"].isna().sum()), "classes": {}, "budgets": list(BUDGETS)}
    for cname, col in CLASSES.items():
        t = test[test[col].notna()]
        y = t[col].to_numpy(float)
        S = {k: (t[k].to_numpy(float), t[tie[k]].to_numpy(float)) for k in names}
        idx = boot_indices(len(t))
        pairs = [("conf_w", "disagree"), ("conf_w", irt_k)] if cname == "primary" else []
        out["classes"][cname] = evaluate_scorers(y, S, idx, pairs)
        if cname == "primary":
            nv = t[t["subject"] != "virology"]
            yv = nv[col].to_numpy(float)
            Sv = {k: (nv[k].to_numpy(float), nv[tie[k]].to_numpy(float)) for k in names}
            out["virology_excluded"] = evaluate_scorers(yv, Sv, boot_indices(len(nv)), pairs)
    pd_ = out["classes"]["primary"]["paired_precision_diff"]
    h1a = all(pd_[k] is not None and pd_[k]["lo"] > 0 for k in pd_)
    out["h1a"] = {"holds": bool(h1a), "rule": "both paired intervals of precision at 10% lie entirely above 0",
                  "contrasts": pd_}
    out["h1b"] = {"computable": False, "reason": "21 multiple-choice weak positives outside GPQA in the Writer release, below 100"}
    cpi = api_cost_per_item()
    n_test = len(test)
    out["cost"] = {"api_usd_per_item": cpi, "review_minutes_per_item_assumed": REVIEW_MINUTES_ASSUMED, "per_scorer": {}}
    prim = out["classes"]["primary"]
    for k in names:
        usd_item = sum(cpi[p] for p in API_SCORER_PURPOSES[k])
        rows = {}
        for b in BUDGETS:
            kk = rank.budget_k(prim["n"], b)
            pr_ = prim["scorers"][k][f"precision_{b}"]
            found = {q: pr_[q] * kk for q in ("est", "lo", "hi")}
            rows[str(b)] = {"errors_found": {"est": found["est"], "lo": found["lo"], "hi": found["hi"]},
                            "usd_per_error": None if found["lo"] <= 0 else
                            {"est": usd_item * n_test / found["est"], "lo": usd_item * n_test / found["hi"], "hi": usd_item * n_test / found["lo"]},
                            "review_minutes_per_error_assumed": None if found["lo"] <= 0 else
                            {"est": REVIEW_MINUTES_ASSUMED * kk / found["est"], "lo": REVIEW_MINUTES_ASSUMED * kk / found["hi"],
                             "hi": REVIEW_MINUTES_ASSUMED * kk / found["lo"]}}
        out["cost"]["per_scorer"][k] = {"api_usd_per_item": usd_item, "budgets": rows}
    out["memorization_exploratory"] = memorization(f)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    from . import report
    report.write_h1(out)
    return out


H2_OUT = paths.RESULTS / "h2.json"


def run_h2() -> dict:
    """H2: lr plus the hidden-state probe against lr alone, primary class, paired precision at 10%."""
    from . import hidden

    commit = prereg_commit()
    blind = unblind(commit)
    f = frame().sort_values("item_id").reset_index(drop=True)
    items_order = pd.read_parquet(paths.ITEMS).sort_values("item_id")["item_id"].tolist()
    assert f["item_id"].tolist() == items_order
    X = np.load(hidden.FEATURES) if hidden.FEATURES.exists() else hidden.features()
    irt_k = select_irt(f)["chosen"]
    feats = available(f, ["disagree", "conf", "conf_w", irt_k, "ptrue_key", "length", "dup_options"])
    base = supervised.fit_score(f, feats, "flaw_primary", "conf_w")
    probe = supervised.fit_probe(X, f, "flaw_primary", "conf_w")
    stacked = supervised.fit_score(f, feats, "flaw_primary", "conf_w", extra=pd.DataFrame({"probe": probe["scores"]}))
    f["lr"], f["probe"], f["lr_probe"] = base["scores"], probe["scores"], stacked["scores"]
    t = f[(f["fold"] == "test") & f["flaw_primary"].notna()]
    y = t["flaw_primary"].to_numpy(float)
    S = {k: (t[k].to_numpy(float), t["conf_w"].to_numpy(float)) for k in ("lr_probe", "lr", "probe", "conf_w")}
    res = evaluate_scorers(y, S, boot_indices(len(t)), [("lr_probe", "lr"), ("probe", "conf_w")])
    d = res["paired_precision_diff"]["lr_probe-lr"]
    meta = json.loads(hidden.FEATURES_META.read_text())
    info = json.loads(scorers.IRT_OUT.read_text())
    out = {"prereg": blind, "amendments": amendments(info["models"]), "lr_features": feats,
           "model": meta["model"], "layer_plan": meta["layer_plan"], "layers": meta["layers"],
           "probe": {k: v for k, v in probe.items() if k != "scores"},
           "stacked_coef": stacked["coef"], "base_C": base["C"], "stacked_C": stacked["C"], "result": res,
           "h2": {"holds": bool(d is not None and d["lo"] > 0),
                  "rule": "paired interval of lr_probe minus lr precision at 10% lies entirely above 0"},
           "clarification": "calibration-fit probe scores are 5-fold out-of-fold by subject, so the stacked model is "
                            "never trained on an in-sample probe score; the preregistration did not specify this"}
    H2_OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out
