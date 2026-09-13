"""Six scorers. Each returns a higher-is-riskier float per question. This module never reads labels.

Cost per scorer is measured from the logged token counts of the extra calls it needs.
length_residual fits mean_logprob ~ length on the calibration split only; coefficients go to
results/calibration.json and are reused for test.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import data
from .cost import cost_usd

EXTRA_CALLS = {"mean_logprob": 0, "min_k_logprob": 0, "length": 0, "length_residual": 0,
               "consistency_k3": 3, "p_true": 1}
K = 5
_sbert = None


def _sbert_model():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer

        name = json.loads((data.CONFIG / "grader.json").read_text())["sbert_model"]
        _sbert = SentenceTransformer(name)
    return _sbert


def mean_logprob(lps: list) -> float:
    return -float(np.mean(lps)) if lps else float("nan")


def min_k_logprob(lps: list, k: int = K) -> float:
    if not lps:
        return float("nan")
    return -float(np.mean(sorted(lps)[:k]))


def consistency(texts: list) -> float:
    """One minus mean pairwise SBERT cosine among the primary answer and its resamples."""
    if len(texts) < 2:
        return float("nan")
    emb = _sbert_model().encode(texts, normalize_embeddings=True)
    sims = emb @ emb.T
    iu = np.triu_indices(len(texts), 1)
    return 1.0 - float(sims[iu].mean())


def p_true(first_top: list) -> float:
    """One minus P(Yes) from the first-token top logprobs of the ptrue call."""
    p_yes = sum(math.exp(t["logprob"]) for t in first_top if t["token"].strip().lower() == "yes")
    return 1.0 - p_yes


def fit_length_residual(scores: pd.DataFrame) -> dict:
    m = scores[["mean_logprob", "length"]].dropna()
    slope, intercept = np.polyfit(m["length"].astype(float), m["mean_logprob"].astype(float), 1)
    return {"slope": float(slope), "intercept": float(intercept), "n": int(len(m))}


def score_split(split: str, length_coef: dict | None = None) -> pd.DataFrame:
    gens = data.read_generations(split)
    cheap = data.models_config()["cheap"]
    rows = []
    for qid, g in gens.groupby("qid", sort=True):
        r = g.set_index("role")
        if "primary" not in r.index:
            continue
        prim = r.loc["primary"]
        lps = json.loads(prim["token_logprobs"])
        row = {"qid": qid, "model": cheap,
               "mean_logprob": mean_logprob(lps), "min_k_logprob": min_k_logprob(lps),
               "length": float(prim["output_tokens"])}
        res = [r.loc[k] for k in ("resample_1", "resample_2", "resample_3") if k in r.index]
        row["consistency_k3"] = consistency([prim["answer_text"]] + [x["answer_text"] for x in res])
        row["p_true"] = p_true(json.loads(r.loc["ptrue"]["token_logprobs"])) if "ptrue" in r.index else float("nan")
        calls = {"mean_logprob": 0, "min_k_logprob": 0, "length": 0, "length_residual": 0,
                 "consistency_k3": len(res), "p_true": int("ptrue" in r.index)}
        cost = {k: 0.0 for k in calls}
        cost["consistency_k3"] = sum(cost_usd(x["model"], x["input_tokens"], x["output_tokens"]) for x in res)
        if "ptrue" in r.index:
            pt = r.loc["ptrue"]
            cost["p_true"] = cost_usd(pt["model"], pt["input_tokens"], pt["output_tokens"])
        row["extra_calls"] = json.dumps(calls)
        row["extra_cost_usd"] = json.dumps(cost)
        rows.append(row)
    df = pd.DataFrame(rows)
    if length_coef is None:
        if split != "calibration":
            raise ValueError("length_residual coefficients must come from the calibration split")
        length_coef = fit_length_residual(df)
        _save_coef(length_coef)
    df["length_residual"] = df["mean_logprob"] - (length_coef["slope"] * df["length"] + length_coef["intercept"])
    cols = ["qid", "model", *data.SCORERS, "extra_calls", "extra_cost_usd"]
    df = df[cols]
    df.to_csv(data.path(split, "scores"), index=False, float_format="%.6f")
    return df


def _save_coef(coef: dict) -> None:
    p = data.RESULTS / "calibration.json"
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["length_residual"] = coef
    p.write_text(json.dumps(cur, indent=2, sort_keys=True))


def load_coef() -> dict:
    p = data.RESULTS / "calibration.json"
    if not p.exists() or "length_residual" not in json.loads(p.read_text()):
        raise FileNotFoundError("score the calibration split first (it fits length_residual)")
    return json.loads(p.read_text())["length_residual"]
