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


class Ranked:
    """One score vector ranked once, with cumulative weights for every resample row of W.

    Ties on the score form blocks. A block straddling a cutoff counts its expected share, as in itemtriage.rank.
    Only the cutoff block and the blocks holding positives are touched, so cost is one cumulative sum per row.
    """

    def __init__(self, s: np.ndarray, W: np.ndarray):
        s = np.asarray(s, float)
        self.order = np.argsort(-s, kind="stable")
        ss = s[self.order]
        new = np.r_[True, ss[1:] != ss[:-1]]
        self.block = np.cumsum(new) - 1
        self.starts = np.flatnonzero(new)
        self.ends = np.r_[self.starts[1:], len(s)] - 1
        self.Wo = W[:, self.order]
        self.cw = np.cumsum(self.Wo, axis=1, dtype=np.float64)
        self.rows = np.arange(len(W))

    def cum_pos(self, y: np.ndarray) -> np.ndarray:
        return np.cumsum(self.Wo * y[self.order], axis=1, dtype=np.float64)

    def precision(self, cp: np.ndarray, budget: float) -> np.ndarray:
        k = float(rank.budget_k(int(round(self.cw[0, -1])), budget))
        i = (self.cw < k).sum(axis=1)                 # first sorted position whose cumulative weight reaches k
        b = self.block[i]
        st, en = self.starts[b], self.ends[b]
        prev = np.maximum(st - 1, 0)
        pw = np.where(st > 0, self.cw[self.rows, prev], 0.0)
        pp = np.where(st > 0, cp[self.rows, prev], 0.0)
        bw, bp = self.cw[self.rows, en] - pw, cp[self.rows, en] - pp
        return (pp + (k - pw) / bw * bp) / k

    def ap(self, cp: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact expected AP under ties in expectation. NaN where a row has no positive."""
        pb = np.unique(self.block[y[self.order] > 0])
        if len(pb) == 0:
            return np.full(len(self.rows), np.nan)
        st, en = self.starts[pb], self.ends[pb]
        f = np.where(st > 0, self.cw[:, np.maximum(st - 1, 0)], 0.0)
        A = np.where(st > 0, cp[:, np.maximum(st - 1, 0)], 0.0)
        m = self.cw[:, en] - f
        p = cp[:, en] - A
        c = np.where(m > 1, (p - 1) / np.where(m > 1, m - 1, 1), 0.0)
        H = digamma(f + m + 1) - digamma(f + 1)
        contrib = np.where(p > 0, p * ((A + 1 - c * (f + 1)) * H + c * m) / np.where(m > 0, m, 1), 0.0)
        P = cp[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(P > 0, contrib.sum(axis=1) / P, np.nan)


def precision_w(s: np.ndarray, y: np.ndarray, W: np.ndarray, budget: float) -> np.ndarray:
    r = Ranked(s, W)
    return r.precision(r.cum_pos(np.asarray(y, float)), budget)


def ap_w(s: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    r = Ranked(s, W)
    y = np.asarray(y, float)
    return r.ap(r.cum_pos(y), y)


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


def with_ones(W: np.ndarray) -> np.ndarray:
    """Prepend a row of ones, so row 0 of every metric is the point estimate on the original items."""
    return np.vstack([np.ones((1, W.shape[1]), W.dtype), W])


class Ctx:
    """Labels and resample weights for one evaluation set. subj None disables virology and within-subject AP."""

    def __init__(self, y: np.ndarray, y2: np.ndarray | None = None, subj: np.ndarray | None = None):
        n, B = len(y), evaluate.N_BOOT
        self.y, self.y2, self.subj = y, y2, subj
        self.W = with_ones(weights(evaluate.boot_indices(n), n))
        self.nv = None if subj is None else subj != "virology"
        if self.nv is not None:
            m = int(self.nv.sum())
            self.Wv = with_ones(weights(evaluate.boot_indices(m), m))
            Ws = weights(strat_index(subj, B), n)
            self.subjects = [g for g in sorted(set(subj)) if y[subj == g].sum() > 0]
            self.masks = {g: subj == g for g in self.subjects}
            self.Ws = {g: with_ones(Ws[:, self.masks[g]]) for g in self.subjects}


def metrics(s: np.ndarray, c: Ctx) -> dict[str, np.ndarray]:
    """Every metric for one score vector. Each array has length 1 + resamples. Row 0 is the point estimate."""
    r = Ranked(s, c.W)
    cp = r.cum_pos(c.y)
    out = {f"precision_{b}": r.precision(cp, b) for b in BUDGETS}
    out["ap"] = r.ap(cp, c.y)
    if c.y2 is not None:
        out["secondary_class_0.1"] = r.precision(r.cum_pos(c.y2), PRIMARY)
    if c.nv is not None:
        rv = Ranked(s[c.nv], c.Wv)
        out["virology_excluded_0.1"] = rv.precision(rv.cum_pos(c.y[c.nv]), PRIMARY)
        per = []
        for g in c.subjects:
            m = c.masks[g]
            rg = Ranked(s[m], c.Ws[g])
            per.append(rg.ap(rg.cum_pos(c.y[m]), c.y[m]))
        A = np.vstack(per)
        out["within_subject_ap"] = np.r_[A[:, 0].mean(), np.nanmean(A[:, 1:], axis=0)]
    return out


def cell_metrics(pids: list[str], S: dict, sc: str, c: Ctx) -> dict[str, tuple]:
    """Nested estimate and replicates per metric for one cell: (point, replicates, draws)."""
    per = [metrics(S[p][sc], c) for p in pids]
    out = {}
    for m in per[0]:
        M = np.vstack([x[m] for x in per])
        pt, reps = nested(M[:, 0], M[:, 1:])
        out[m] = (pt, reps, len(pids))
    return out


def valid(pids: list[str], S: dict, sc: str) -> list[str]:
    return [p for p in pids if sc in S[p] and np.all(np.isfinite(S[p][sc]))]


def score_all(plist: list[dict], inp: dict, rows: np.ndarray, use_irt: bool = True) -> dict[str, dict[str, np.ndarray]]:
    """Scores restricted to `rows` (indices into the matrix items) for every panel and scorer."""
    irt_d, theta = panels.irt_scores() if use_irt else ({}, {})
    ids = inp["item_id"][rows]
    out = {}
    for p in plist:
        s = {k: v[rows] for k, v in panels.cheap_scores(p, inp, theta.get(p["id"])).items()}
        if p["id"] in irt_d:
            s["irt_small"] = irt_d[p["id"]].reindex(ids).to_numpy()
        out[p["id"]] = s
    return out


_POOL: dict = {}


def _scorer_job(sc: str) -> tuple[str, dict]:
    S, cells, c = _POOL["S"], _POOL["cells"], _POOL["ctx"]
    res = {}
    for key, pids in cells.items():
        pids = valid(pids, S, sc)
        if pids:
            res[key] = cell_metrics(pids, S, sc, c)
    return sc, res


def all_cells(S: dict, cells: dict, c: Ctx, scorer_list, workers: int) -> dict:
    """scorer -> cell -> metric -> (point, replicates, draws). Scorers run in parallel worker processes."""
    from multiprocessing import get_context

    _POOL.update(S=S, cells=cells, ctx=c)
    with get_context("fork").Pool(min(workers, len(scorer_list))) as pool:
        return dict(pool.map(_scorer_job, scorer_list))


def cells_of(plist: list[dict]) -> dict:
    c = {}
    for p in plist:
        c.setdefault((p["stratum"], p["k"]), []).append(p["id"])
    return c


def table(res: dict, metric: str) -> list[dict]:
    rows = []
    for (s, k), mm in sorted(res.items(), key=lambda x: (x[0][0], x[0][1])):
        pt, reps, n = mm[metric]
        rows.append({"stratum": s, "k": k, "draws": n, **ci(pt, reps)})
    return rows


def diff(res: dict, a, b, metric: str = f"precision_{PRIMARY}") -> dict | None:
    if a not in res or b not in res:
        return None
    (pa, ra, _), (pb, rb, _) = res[a][metric], res[b][metric]
    d = ci(pa - pb, ra - rb)
    d["share_resamples_at_or_below_zero"] = float(np.mean((ra - rb) <= 0))
    return d


def paired_draws(pids: list[str], S: dict, a: str, b: str, c: Ctx) -> dict:
    """Scorer a minus scorer b on the same panel draws, precision at 10%, nested over items and draws."""
    D = np.vstack([metrics(S[p][a], c)[f"precision_{PRIMARY}"] - metrics(S[p][b], c)[f"precision_{PRIMARY}"] for p in pids])
    pt, reps = nested(D[:, 0], D[:, 1:])
    d = ci(pt, reps)
    d["share_resamples_at_or_below_zero"] = float(np.mean(reps <= 0))
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


def local_point(ids: np.ndarray, c: Ctx) -> dict:
    sc = pd.read_parquet(scorers.SCORES).set_index("item_id").loc[ids]
    out = {"models": scorers.panel_models(), "regime": "zero-shot, direct letters, own run (Study 1)", "k": len(scorers.panel_models())}
    for s in LOCAL:
        m = metrics(sc[s].to_numpy(float), c)[f"precision_{PRIMARY}"]
        out[s] = ci(m[0], m[1:])
    return out


def gsm8k(reps: pd.DataFrame, plist: list[dict], workers: int) -> dict:
    """H4 confirmatory and secondary metrics on platinum GSM8K."""
    lab = labels_mod.load_gsm8k(caller="itemtriage.study2")
    corr = pd.read_parquet(matrix.GSM_CORRECT)
    q = pd.read_parquet(matrix.GSM_ITEMS).set_index("item_id").loc[corr["item_id"], "question"]
    inp = {"item_id": corr["item_id"].to_numpy(), "subject": np.array(["gsm8k"] * len(corr)),
           "Y": corr.drop(columns="item_id"), "P": None, "length": q.str.len().to_numpy(float)}
    pos = {i: n for n, i in enumerate(inp["item_id"])}
    rows = np.array([pos[i] for i in lab["item_id"]])
    y = lab["flaw"].to_numpy(float)
    S = {p["id"]: {k: v for k, v in score_all([p], inp, rows, use_irt=False)[p["id"]].items() if k in GSM_SCORERS} for p in plist}
    c = Ctx(y)
    cells = cells_of(plist)
    res = all_cells(S, cells, c, [s for s in GSM_SCORERS if s != "length"], workers)
    L = metrics(inp["length"][rows], c)
    out = {"items": len(y), "positives": int(y.sum()), "flagged_at_10": rank.budget_k(len(y), PRIMARY),
           "curves": {sc: {m: table(r, m) for m in ("precision_0.05", "precision_0.1", "precision_0.2", "ap")} for sc, r in res.items()},
           "length_control": {m: ci(v[0], v[1:]) for m, v in L.items()}}
    d = diff(res["disagree"], *H4_PAIR)
    out["h4"] = {"contrast": d, "verdict": verdict(d), "role": "confirmatory", "scorer": "disagree"}
    return out


def run(workers: int = 8) -> dict:
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
    c = Ctx(y, y2, t["subject"].to_numpy())
    S = score_all(plist, inp, rows)
    cells = cells_of(plist)
    res = all_cells(S, cells, c, [s for s in SCORERS if s != "length"], workers)
    L = metrics(inp["length"][rows], c)
    all_key = ("all", len(reps))
    metric_names = list(next(iter(next(iter(res.values())).values())).keys())
    out = {"prereg_commit": commit, "label": "MMLU-Redux results are exploratory and post-unblinding (PREREG_2).",
           "items": len(t), "positives": int(y.sum()), "models": len(reps), "panels": len(plist),
           "p_key_models": 0 if inp["P"] is None else int(inp["P"].shape[1]),
           "within_subject_ap_subjects": len(c.subjects),
           "curves": {sc: {m: table(r, m) for m in metric_names} for sc, r in res.items()},
           "length_control": {m: ci(v[0], v[1:]) for m, v in L.items()}}

    d = diff(res["disagree"], *H4_PAIR)
    out["h4_mmlu"] = {"contrast": d, "verdict": verdict(d), "role": "exploratory, post-unblinding", "scorer": "disagree"}
    P10 = f"precision_{PRIMARY}"
    allp = res["disagree"][all_key][P10][0]
    curve = {k: ci(mm[P10][0], mm[P10][1]) for (st, k), mm in res["disagree"].items() if st in ("random", "all")}
    ks = sorted(curve)
    ok = [k for k in ks if all(curve[j]["lo"] >= H5_SHARE * allp for j in ks if j >= k)]
    out["h5"] = {"scorer": "disagree", "all_model_estimate": allp, "target": H5_SHARE * allp,
                 "smallest_k": ok[0] if ok else None, "curve_at_k": curve[ok[0]] if ok else None}
    h6 = {}
    for sc in ("irt_small", "item_total"):
        rows6 = []
        for k in [*panels.IRT_SIZES, len(reps)]:
            pids = valid(cells[("all", k) if k == len(reps) else ("random", k)], S, sc)
            pids = valid(pids, S, "disagree")
            if pids:
                rows6.append({"k": k, "draws": len(pids), **paired_draws(pids, S, sc, "disagree", c)})
        cross = [r["k"] for r in rows6 if r["lo"] > 0]
        h6[sc] = {"rows": rows6, "crossing_k": cross[0] if cross else None}
    out["h6"] = h6
    out["local_point"] = local_point(t["item_id"].to_numpy(), c)
    out["params_proxy"] = {k: v for k, v in params_proxy(plist).items() if k.split("-")[0] in ("top", "bottom", "all")}
    out["memorization_by_decile"] = {"calibration_and_validation": memorization_by_decile(f, inp, reps, ("calibration_fit", "validation")),
                                     "test_post_unblinding": memorization_by_decile(f, inp, reps, ("test",))}
    out["gsm8k"] = gsm8k(reps, plist, workers)
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
        rows = o["curves"].get(sc, {}).get(f"precision_{PRIMARY}", [])
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
