"""Shared loaders and statistics for the HalluScope audit (eval/audit/).

CPU only. Reads the committed result CSVs; nothing is simulated. Two derived
inputs are computed once and cached under eval/audit/out/:
  * token_counts.csv   - answer lengths re-tokenized from the logged answer text
                         (needs `tiktoken` and the cached TinyLlama tokenizer)
  * gold_sbert_cos.csv - SBERT cosines for the gold set (same model as eval/match.py)
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only; never needs gpu.lock
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent.parent
OUT = AUDIT / "out"
FIG = AUDIT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "halluscope-core" / "src"))  # when not pip-installed

from eval.stats import delong_test  # noqa: E402
from judgements import JUDGE  # noqa: E402

N_BOOT = 2000
SEED = 42
SBERT_THR = 0.35
SUPPRESSED_PREFIX = "⚠️ **High hallucination risk detected.**"
GOLD_CSV = ROOT / "eval" / "gold" / "gold_results_api.csv"
RESULT_CSV = {m: ROOT / "eval" / f"results_{m}.csv" for m in ("api", "local")}


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── derived inputs (cached) ──────────────────────────────────────────
def token_counts():
    cache = OUT / "token_counts.csv"
    if not cache.exists():
        import tiktoken
        from transformers import AutoTokenizer

        enc = tiktoken.get_encoding("o200k_base")  # gpt-4o-mini tokenizer
        tl = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        count = {
            "api": lambda t: len(enc.encode(t)),
            "gold": lambda t: len(enc.encode(t)),
            "local": lambda t: len(tl(t, add_special_tokens=False)["input_ids"]),
        }
        paths = {**RESULT_CSV, "gold": GOLD_CSV}
        with open(cache, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["set", "id", "tokens"])
            for s, p in paths.items():
                for r in read_csv(p):
                    a = r["model_answer"]
                    n = "" if a.startswith(SUPPRESSED_PREFIX) else count[s](a)
                    w.writerow([s, r["id"], n])
    return {
        (r["set"], r["id"]): (int(r["tokens"]) if r["tokens"] else None)
        for r in read_csv(cache)
    }


def gold_cosines():
    cache = OUT / "gold_sbert_cos.csv"
    if not cache.exists():
        from eval.match import batch_cosine

        rows = read_csv(GOLD_CSV)
        sims = batch_cosine(
            [r["model_answer"] for r in rows], [r["reference_answer"] for r in rows]
        )
        with open(cache, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "sbert_cos"])
            w.writerows((r["id"], f"{s:.4f}") for r, s in zip(rows, sims))
    return {r["id"]: float(r["sbert_cos"]) for r in read_csv(cache)}


# ── loaders ──────────────────────────────────────────────────────────
def _tag(rows, set_name):
    toks = token_counts()
    for r in rows:
        r["hri_f"] = float(r["hri"])
        r["suppressed"] = r["model_answer"].startswith(SUPPRESSED_PREFIX)
        r["judge"] = JUDGE[set_name][r["id"]]
        r["ya"] = int(r["judge"] == "I")  # audit label, meaningful only if judge != "U"
        r["tokens"] = toks.get((set_name, r["id"]))
    return sorted(rows, key=lambda r: r["id"])


def load_mode(mode):
    """Committed eval/results_{mode}.csv. y = SBERT label as reported (1 = mismatch)."""
    rows = read_csv(RESULT_CSV[mode])
    for r in rows:
        r["cos"] = float(r["sbert_cos"])
        r["y"] = int(r["model_match"] == "mismatch")
    return _tag(rows, mode)


def load_gold():
    """eval/gold/gold_results_api.csv, labelled here with SBERT at 0.35."""
    rows = read_csv(GOLD_CSV)
    cos = gold_cosines()
    for r in rows:
        r["cos"] = cos[r["id"]]
        r["y"] = int(r["cos"] < SBERT_THR)
    return _tag(rows, "gold")


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def raw_from_hri(h):
    """Invert HRI = sigmoid(10*raw - 3) (heuristic weights). NaN where 4-dp rounding breaks it."""
    if not 0.0005 < h < 0.9995:
        return float("nan")
    return (math.log(h / (1 - h)) + 3.0) / 10.0


# ── statistics ───────────────────────────────────────────────────────
def _auc(y, s):
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def boot(stat, *arrays):
    """Percentile bootstrap (N_BOOT resamples, seed 42). Degenerate resamples are skipped."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        v = stat(*(a[idx] for a in arrays))
        if v is not None and np.isfinite(v):
            vals.append(v)
    est = stat(*arrays)
    if est is None or not np.isfinite(est) or not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n": n,
            "n_boot_valid": len(vals)}


def auc_ci(y, s):
    d = boot(_auc, y, s)
    if d:
        y = np.asarray(y)
        d.update(pos=int(y.sum()), neg=int(len(y) - y.sum()), width=d["hi"] - d["lo"],
                 contains_half=bool(d["lo"] <= 0.5 <= d["hi"]))
    return d


def spearman_ci(x, y):
    return boot(lambda a, b: float(sps.spearmanr(a, b).statistic), x, y)


def kappa_ci(a, b):
    return boot(lambda p, q: float(cohen_kappa_score(p, q)), a, b)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"est": p, "lo": c - h, "hi": c + h, "k": int(k), "n": int(n)}


def delong_diff(y, a, b):
    """DeLong p-value (eval/stats.py) plus a paired bootstrap CI on AUROC(a) - AUROC(b)."""
    auc_a, auc_b, p = delong_test(np.asarray(y), np.asarray(a), np.asarray(b))
    diff = boot(lambda yy, aa, bb: None if len(np.unique(yy)) < 2
                else _auc(yy, aa) - _auc(yy, bb), y, a, b)
    return {"auc_a": auc_a, "auc_b": auc_b, "p_delong": p, "diff": diff}


def f(d, k=3):
    if d is None:
        return "n/a"
    return f"{d['est']:.{k}f} [{d['lo']:.{k}f}, {d['hi']:.{k}f}]"


def save(name, obj):
    def _default(o):
        return o.item() if hasattr(o, "item") else str(o)

    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_default))
