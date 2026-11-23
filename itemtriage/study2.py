"""Study 2 evaluation (docs/PREREG_2.md). Labels join here only.

Precision and average precision are computed for all 2,000 bootstrap resamples at once: each resample is a
vector of item multiplicities, and ties resolve in expectation exactly as in itemtriage.rank on the expanded data.
Cells with several panel draws use a nested bootstrap: items outer, draws inner. Writes results/itemtriage/study2.json
and figures/itemtriage/panel_size.png.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.special import digamma

from . import evaluate, matrix, panels, paths, rank, scorers, splits
from . import labels as labels_mod

OUT = paths.RESULTS / "study2.json"
FIG = paths.FIGURES / "panel_size.png"
PREREG = "docs/PREREG_2.md"
BUDGETS = (0.05, 0.10, 0.20)
PRIMARY = 0.10
H5_SHARE = 0.80
H4_PAIR = (("top", 3), ("random", 10))
SCORERS = ("disagree", "strong3", "item_total", "irt_small", "subject_mean", "conf", "conf_w", "length")
GSM_SCORERS = ("disagree", "strong3", "item_total", "length")
LOCAL = ("disagree", "conf_w")
CI = (2.5, 97.5)


# ------------------------------------------------------------------ weighted metrics over resamples
def weights(idx: list[np.ndarray], n: int) -> np.ndarray:
    """Resample index lists -> multiplicity matrix (resamples x items)."""
    return np.vstack([np.bincount(ix, minlength=n) for ix in idx]).astype(np.int32)


def _blocks(s: np.ndarray):
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    return order, starts


def precision_w(s: np.ndarray, y: np.ndarray, W: np.ndarray, budget: float) -> np.ndarray:
    """Expected precision of the top ceil(budget n) under ties in expectation, for each row of W (n = row sum)."""
    order, starts = _blocks(np.asarray(s, float))
    Wo = W[:, order]
    bw = np.add.reduceat(Wo, starts, axis=1).astype(float)
    bp = np.add.reduceat(Wo * y[order], starts, axis=1).astype(float)
    k = float(rank.budget_k(int(W[0].sum()), budget))   # every resample has the original size
    cw, cp = np.cumsum(bw, axis=1), np.cumsum(bp, axis=1)
    b = (cw < k).sum(axis=1)                             # first block whose cumulative weight reaches k
    rows = np.arange(len(W))
    prev_w = np.where(b > 0, cw[rows, np.maximum(b - 1, 0)], 0.0)
    prev_p = np.where(b > 0, cp[rows, np.maximum(b - 1, 0)], 0.0)
    frac = np.where(bw[rows, b] > 0, (k - prev_w) / np.where(bw[rows, b] > 0, bw[rows, b], 1.0), 0.0)
    return (prev_p + frac * bp[rows, b]) / k


def ap_w(s: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Exact expected average precision under ties in expectation, per row of W. NaN where a row has no positive."""
    order, starts = _blocks(np.asarray(s, float))
    Wo = W[:, order].astype(float)
    m = np.add.reduceat(Wo, starts, axis=1)
    p = np.add.reduceat(Wo * y[order], starts, axis=1)
    f = np.cumsum(m, axis=1) - m
    A = np.cumsum(p, axis=1) - p
    c = np.where(m > 1, (p - 1) / np.where(m > 1, m - 1, 1), 0.0)
    H = digamma(f + m + 1) - digamma(f + 1)
    contrib = np.where(m > 0, p * ((A + 1 - c * (f + 1)) * H + c * m) / np.where(m > 0, m, 1), 0.0)
    P = p.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(P > 0, contrib.sum(axis=1) / P, np.nan)


# ------------------------------------------------------------------ nested cells
def ci(point: float, reps: np.ndarray) -> dict:
    r = reps[np.isfinite(reps)]
    lo, hi = np.percentile(r, CI) if len(r) else (np.nan, np.nan)
    return {"est": float(point), "lo": float(lo), "hi": float(hi), "n_boot_valid": int(len(r))}


def inner_index(R: int, B: int) -> np.ndarray:
    """Resampled draw indices per outer replicate (B x R), from a fixed seed."""
    return np.random.RandomState(evaluate.SEED).randint(0, R, (B, R))


def nested(point_draws: np.ndarray, boot_draws: np.ndarray) -> tuple[float, np.ndarray]:
    """point_draws: R. boot_draws: R x B. Returns the draw mean and B nested replicates."""
    R, B = boot_draws.shape
    if R == 1:
        return float(point_draws[0]), boot_draws[0]
    G = inner_index(R, B)
    return float(np.mean(point_draws)), boot_draws[G, np.arange(B)[:, None]].mean(axis=1)


# ------------------------------------------------------------------ assembly
def eval_frame() -> pd.DataFrame:
    """Matched test-fold items with known primary labels, plus the secondary class."""
    corr_ids = pd.read_parquet(matrix.CORRECT, columns=["item_id"])["item_id"]
    lab = labels_mod.load(caller="itemtriage.study2")[["item_id", "flaw_primary", "flaw_any"]]
    fold = splits.fold_of(splits.load())
    f = pd.DataFrame({"item_id": corr_ids})
    f["fold"] = f["item_id"].map(fold)
    f = f.merge(lab, on="item_id", how="left")
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")["subject"]
    f["subject"] = f["item_id"].map(items)
    return f


def test_ids() -> np.ndarray:
    corr_ids = pd.read_parquet(matrix.CORRECT, columns=["item_id"])["item_id"]
    fold = splits.fold_of(splits.load())
    return np.array(sorted(i for i in corr_ids if fold[i] == "test"))


def strat_index(subjects: np.ndarray, B: int) -> list[np.ndarray]:
    """Item resamples drawn within each subject (stratified), seed 42."""
    rng = np.random.RandomState(evaluate.SEED)
    groups = [np.flatnonzero(subjects == s) for s in sorted(set(subjects))]
    return [np.concatenate([g[rng.randint(0, len(g), len(g))] for g in groups]) for _ in range(B)]


def within_subject_ap(s: np.ndarray, y: np.ndarray, subj: np.ndarray, Ws: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Mean within-subject AP over subjects with a positive. Returns point, per-resample values, skipped subject-resamples."""
    keep = [g for g in sorted(set(subj)) if y[subj == g].sum() > 0]
    pts, reps = [], []
    for g in keep:
        m = subj == g
        pts.append(rank.average_precision(y[m], s[m], np.zeros(m.sum())))
        reps.append(ap_w(s[m], y[m], Ws[:, m]))
    R = np.vstack(reps)
    return float(np.mean(pts)), np.nanmean(R, axis=0), int(np.isnan(R).sum())


def score_all(plist: list[dict], inp: dict, rows: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Scores restricted to `rows` (indices into the matrix items) for every panel and scorer."""
    irt_d, theta = panels.irt_scores()
    ids = inp["item_id"][rows]
    out = {}
    for p in plist:
        s = {k: v[rows] for k, v in panels.cheap_scores(p, inp, theta.get(p["id"])).items()}
        if p["id"] in irt_d:
            s["irt_small"] = irt_d[p["id"]].reindex(ids).to_numpy()
        s["length"] = inp["length"][rows]
        out[p["id"]] = s
    return out


def cell_stats(cells: dict, S: dict, y: np.ndarray, W: np.ndarray, budget: float, scorer: str, metric: str = "precision"):
    """For each (stratum, k): nested point and replicates of one scorer's metric. Skips cells without that scorer."""
    res = {}
    for key, pids in cells.items():
        pids = [p for p in pids if scorer in S[p] and np.all(np.isfinite(S[p][scorer]))]
        if not pids:
            continue
        if metric == "precision":
            pt = np.array([rank.precision_at(y, S[p][scorer], np.zeros(len(y)), budget) for p in pids])
            bt = np.vstack([precision_w(S[p][scorer], y, W, budget) for p in pids])
        else:
            pt = np.array([rank.average_precision(y, S[p][scorer], np.zeros(len(y))) for p in pids])
            bt = np.vstack([ap_w(S[p][scorer], y, W) for p in pids])
        res[key] = (*nested(pt, bt), len(pids))
    return res


def cells_of(plist: list[dict]) -> dict:
    c = {}
    for p in plist:
        c.setdefault((p["stratum"], p["k"]), []).append(p["id"])
    return c


def table(res: dict) -> list[dict]:
    return [{"stratum": s, "k": k, "draws": n, **ci(pt, reps)} for (s, k), (pt, reps, n) in sorted(res.items(), key=lambda x: (x[0][0], x[0][1]))]


def diff(res: dict, a, b) -> dict | None:
    if a not in res or b not in res:
        return None
    d = ci(res[a][0] - res[b][0], res[a][1] - res[b][1])
    d["share_resamples_at_or_below_zero"] = float(np.mean((res[a][1] - res[b][1]) <= 0))
    return d


def verdict(d: dict | None) -> str:
    if d is None:
        return "not computable"
    return "holds" if d["lo"] > 0 else "reversed" if d["hi"] < 0 else "inconclusive"


def params_proxy(plist: list[dict]) -> dict:
    """Estimated total parameters per panel: k times the median known member count. A labelled proxy."""
    m = pd.read_parquet(matrix.MODELS).set_index("model")["params"]
    out = {}
    for p in plist:
        known = m.loc[p["models"]].dropna()
        out[p["id"]] = {"estimate": float(p["k"] * known.median()) if len(known) else None, "share_known": len(known) / p["k"]}
    return out


def memorization_by_decile(f: pd.DataFrame, inp: dict, reps: pd.DataFrame, folds: tuple) -> list[dict]:
    """Mean agreement with the key on flawed and sound items per ability decile, with item-bootstrap intervals."""
    sub = f[f["fold"].isin(folds) & f["flaw_primary"].notna()]
    pos = {i: n for n, i in enumerate(inp["item_id"])}
    rows = np.array([pos[i] for i in sub["item_id"]])
    y = sub["flaw_primary"].to_numpy(int)
    dec = pd.qcut(reps["ability"].rank(method="first", ascending=True), 10, labels=False).to_numpy()
    out = []
    for d in range(10):
        Y = inp["Y"][reps["model"][dec == d].tolist()].to_numpy(float)[rows].mean(axis=1)
        for cls, mask in (("flawed", y == 1), ("sound", y == 0)):
            v = Y[mask]
            idx = evaluate.boot_indices(len(v))
            out.append({"decile": d + 1, "class": cls, "items": int(mask.sum()),
                        "ability_range": [float(reps["ability"][dec == d].min()), float(reps["ability"][dec == d].max())],
                        **ci(v.mean(), np.array([v[i].mean() for i in idx]))})
    return out


def local_point(f: pd.DataFrame, rows_ids: np.ndarray, y: np.ndarray, W: np.ndarray) -> dict:
    sc = pd.read_parquet(scorers.SCORES).set_index("item_id").loc[rows_ids]
    out = {"models": scorers.panel_models(), "regime": "zero-shot, direct letters, own run (Study 1)", "k": len(scorers.panel_models())}
    for s in LOCAL:
        v = sc[s].to_numpy(float)
        out[s] = ci(rank.precision_at(y, v, np.zeros(len(y)), PRIMARY), precision_w(v, y, W, PRIMARY))
    return out


def gsm8k(reps: pd.DataFrame) -> dict:
    """H4 confirmatory and secondary metrics on platinum GSM8K."""
    lab = labels_mod.load_gsm8k(caller="itemtriage.study2")
    corr = pd.read_parquet(matrix.GSM_CORRECT)
    q = pd.read_parquet(matrix.GSM_ITEMS).set_index("item_id").loc[corr["item_id"], "question"]
    inp = {"item_id": corr["item_id"].to_numpy(), "subject": np.array(["gsm8k"] * len(corr)),
           "Y": corr.drop(columns="item_id"), "P": None, "length": q.str.len().to_numpy(float)}
    plist = panels.draw_panels(reps)
    pos = {i: n for n, i in enumerate(inp["item_id"])}
    rows = np.array([pos[i] for i in lab["item_id"]])
    y = lab["flaw"].to_numpy(float)
    S = {}
    for p in plist:
        s = {k: v[rows] for k, v in panels.cheap_scores(p, inp, None).items() if k in GSM_SCORERS}
        s["length"] = inp["length"][rows]
        S[p["id"]] = s
    W = weights(evaluate.boot_indices(len(y)), len(y))
    cells = cells_of(plist)
    out = {"items": len(y), "positives": int(y.sum()), "flagged_at_10": rank.budget_k(len(y), PRIMARY), "curves": {}}
    for sc in GSM_SCORERS:
        res = cell_stats(cells, S, y, W, PRIMARY, sc)
        out["curves"][sc] = table(res)
        if sc == "disagree":
            d = diff(res, *H4_PAIR)
            out["h4"] = {"contrast": d, "verdict": verdict(d), "role": "confirmatory"}
    return out


def run() -> dict:
    commit = evaluate._git("log", "-1", "--format=%H", "--", PREREG).stdout.strip()
    if not commit or evaluate._git("diff", "--quiet", "HEAD", "--", PREREG).returncode:
        raise SystemExit(f"{PREREG} must be committed and unmodified before Study 2 metrics")
    reps = panels.representatives()
    plist = panels.draw_panels(reps)
    inp = panels.matrix_inputs()
    f = eval_frame()
    t = f[(f["fold"] == "test") & f["flaw_primary"].notna()].reset_index(drop=True)
    pos = {i: n for n, i in enumerate(inp["item_id"])}
    rows = np.array([pos[i] for i in t["item_id"]])
    y, y2 = t["flaw_primary"].to_numpy(float), t["flaw_any"].to_numpy(float)
    subj = t["subject"].to_numpy()
    B = evaluate.N_BOOT
    W = weights(evaluate.boot_indices(len(t)), len(t))
    nv = subj != "virology"
    Wv = weights(evaluate.boot_indices(int(nv.sum())), int(nv.sum()))
    Ws = weights(strat_index(subj, B), len(t))
    S = score_all(plist, inp, rows)
    cells = cells_of(plist)

    out = {"prereg_commit": commit, "label": "MMLU-Redux results are exploratory and post-unblinding (PREREG_2).",
           "items": len(t), "positives": int(y.sum()), "models": len(reps), "panels": len(plist),
           "p_key_models": 0 if inp["P"] is None else int(inp["P"].shape[1]), "curves": {}, "secondary": {}}
    prim = {}
    for sc in SCORERS:
        res = cell_stats(cells, S, y, W, PRIMARY, sc)
        prim[sc] = res
        out["curves"][sc] = table(res)
        sec = {"precision_0.05": table(cell_stats(cells, S, y, W, 0.05, sc)),
               "precision_0.2": table(cell_stats(cells, S, y, W, 0.20, sc)),
               "secondary_class_0.1": table(cell_stats(cells, S, y2, W, PRIMARY, sc)),
               "virology_excluded_0.1": table(cell_stats(cells, {p: {k: v[nv] for k, v in s.items()} for p, s in S.items()},
                                                         y[nv], Wv, PRIMARY, sc)),
               "ap": table(cell_stats(cells, S, y, W, PRIMARY, sc, metric="ap"))}
        ws = {}
        for key, pids in cells.items():
            pids = [p for p in pids if sc in S[p] and np.all(np.isfinite(S[p][sc]))]
            if not pids:
                continue
            r = [within_subject_ap(S[p][sc], y, subj, Ws) for p in pids]
            pt, reps_ = nested(np.array([x[0] for x in r]), np.vstack([x[1] for x in r]))
            ws[key] = (pt, reps_, len(pids))
        sec["within_subject_ap"] = table(ws)
        out["secondary"][sc] = sec

    d = diff(prim["disagree"], *H4_PAIR)
    out["h4_mmlu"] = {"contrast": d, "verdict": verdict(d), "role": "exploratory, post-unblinding"}
    allp = prim["disagree"][("all", len(reps))][0]
    curve = {k: ci(pt, r) for (s, k), (pt, r, _) in prim["disagree"].items() if s == "random"}
    curve[len(reps)] = ci(allp, prim["disagree"][("all", len(reps))][1])
    ks = sorted(curve)
    ok = [k for k in ks if all(curve[j]["lo"] >= H5_SHARE * allp for j in ks if j >= k)]
    out["h5"] = {"all_model_estimate": allp, "target": H5_SHARE * allp, "smallest_k": ok[0] if ok else None,
                 "curve_at_k": curve[ok[0]] if ok else None}
    h6 = {}
    for sc in ("irt_small", "item_total"):
        rows6 = []
        for k in [*panels.IRT_SIZES, len(reps)]:
            stratum = "all" if k == len(reps) else "random"
            pids = cells[(stratum, k)]
            pids = [p for p in pids if sc in S[p] and np.all(np.isfinite(S[p][sc]))]
            if not pids:
                continue
            a = cell_stats({"a": pids}, S, y, W, PRIMARY, sc)["a"]
            b = cell_stats({"b": pids}, S, y, W, PRIMARY, "disagree")["b"]
            dd = ci(a[0] - b[0], a[1] - b[1])
            rows6.append({"k": k, "draws": len(pids), **dd})
        cross = [r["k"] for r in rows6 if r["lo"] > 0]
        h6[sc] = {"rows": rows6, "crossing_k": cross[0] if cross else None}
    out["h6"] = h6
    out["local_point"] = local_point(f, t["item_id"].to_numpy(), y, W)
    out["params_proxy"] = {k: v for k, v in params_proxy(plist).items() if k.endswith("-0") or k.startswith("all")}
    out["memorization_by_decile"] = {"calibration_and_validation": memorization_by_decile(f, inp, reps, ("calibration_fit", "validation")),
                                     "test_post_unblinding": memorization_by_decile(f, inp, reps, ("test",))}
    out["gsm8k"] = gsm8k(reps)
    out["within_subject_ap_subjects"] = int(sum(1 for g in set(subj) if y[subj == g].sum() > 0))
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=_jsonable))
    figure(out)
    return out


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if math.isnan(x) else float(x)
    raise TypeError(type(x))


def figure(o: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"top": ("#1b6ca8", "o"), "bottom": ("#c0392b", "v"), "random": ("#555555", "s"), "lineage": ("#2e8b57", "D")}
    n_all = o["models"]
    for ax, sc, title in ((axes[0], "disagree", "disagree"), (axes[1], "irt_small", "irt_small (Δℓ)")):
        rows = o["curves"].get(sc, [])
        for st, (col, mk) in style.items():
            r = sorted((x for x in rows if x["stratum"] == st), key=lambda x: x["k"])
            if not r:
                continue
            k = [x["k"] for x in r]
            ax.plot(k, [100 * x["est"] for x in r], marker=mk, color=col, label=st)
            ax.fill_between(k, [100 * x["lo"] for x in r], [100 * x["hi"] for x in r], color=col, alpha=0.12)
        a = [x for x in rows if x["stratum"] == "all"]
        if a:
            ax.errorbar([n_all], [100 * a[0]["est"]], yerr=[[100 * (a[0]["est"] - a[0]["lo"])], [100 * (a[0]["hi"] - a[0]["est"])]],
                        fmt="*", color="black", ms=11, label=f"all {n_all}")
        lp = o["local_point"]["disagree"]
        ax.errorbar([o["local_point"]["k"]], [100 * lp["est"]], yerr=[[100 * (lp["est"] - lp["lo"])], [100 * (lp["hi"] - lp["est"])]],
                    fmt="P", color="#8e44ad", ms=10, label="own 5-model panel (zero-shot letters, disagree)")
        ax.set_xscale("log")
        ax.set_xlabel("panel size k (leaderboard models, 5-shot log-likelihood)")
        ax.set_ylabel("precision at 10% budget (%)")
        ax.set_title(f"MMLU-Redux test, {title}. Exploratory, post-unblinding")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    plt.close(fig)
