"""Validity gates on the test split, adapted from the v1 audit (git tag v1, eval/audit/).

Gates: template contamination, split disjointness, minimum n for AUROC 0.65 at 80 percent power,
and per scorer AUROC with interval plus DeLong tests against length and against mean_logprob.
Writes results/gates_test.md and results/gates_test.json. Labels are joined here only for evaluation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data, stats
from .generate import WARNING_BANNER

P_FLAG = 0.05


def scored_labelled(split: str, model_role: str = "cheap") -> pd.DataFrame:
    """Scores joined with consensus labels for the given model. AMBIGUOUS rows dropped. y=1 is INCORRECT."""
    d = data.load_split(split)
    s = data.read_scores(split)
    model = data.models_config()[model_role]
    c = d["consensus"]
    c = c[c["model"] == model][["qid", "label"]]
    m = s.merge(c, on="qid", how="inner")
    m["ambiguous"] = m["label"] == "AMBIGUOUS"
    return m


def run(split: str = "test") -> dict:
    d = data.load_split(split)
    gens = d["generations"]
    m = scored_labelled(split)
    n_amb = int(m["ambiguous"].sum())
    m = m[~m["ambiguous"]].copy()
    y = (m["label"] == "INCORRECT").astype(int).to_numpy()
    out = {"split": split, "n_scored": int(len(m)), "n_ambiguous_excluded": n_amb,
           "positives": int(y.sum()), "negatives": int(len(y) - y.sum())}
    out["template_rows"] = int(gens["answer_text"].str.startswith(WARNING_BANNER).sum())
    data.assert_disjoint()
    out["disjoint"] = True
    prev = y.mean() if len(y) else 0.0
    out["power_at_actual_n"] = stats.auc_power(int(y.sum()), int(len(y) - y.sum())) if 0 < y.sum() < len(y) else None
    out["n_required_for_power_0.80"] = stats.n_required(prev) if 0 < prev < 1 else None
    out["scorers"] = {}
    for sc in data.SCORERS:
        s = m[sc].astype(float).to_numpy()
        ok = np.isfinite(s)
        rec = {"n": int(ok.sum()), "auroc": stats.auc_ci(y[ok], s[ok])}
        for ref in ("length", "mean_logprob"):
            if ref == sc:
                continue
            r = m[ref].astype(float).to_numpy()
            ok2 = ok & np.isfinite(r)
            rec[f"vs_{ref}"] = stats.delong(y[ok2], s[ok2], r[ok2])
        rec["beats_length"] = bool(sc != "length" and rec["vs_length"]["p_delong"] < P_FLAG
                                   and rec["vs_length"]["auc_a"] > rec["vs_length"]["auc_b"])
        out["scorers"][sc] = rec
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    (data.RESULTS / f"gates_{split}.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    (data.RESULTS / f"gates_{split}.md").write_text(render(out))
    return out


def flags(split: str = "test") -> dict:
    """Scorer -> True if not distinguishable from length. Used by every later figure."""
    p = data.RESULTS / f"gates_{split}.json"
    g = json.loads(p.read_text())
    return {sc: (sc == "length" or not rec["beats_length"]) for sc, rec in g["scorers"].items()}


def render(g: dict) -> str:
    L = [f"# Validity gates on the {g['split']} split", "",
         "Brackets are 95 percent bootstrap intervals (2000 resamples, seed 42). Positives are INCORRECT rows.", "",
         "## Gates", "",
         f"- Template contamination: {g['template_rows']} answers match the v1 warning banner "
         f"({'PASS' if g['template_rows'] == 0 else 'FAIL'}).",
         f"- Split disjointness: calibration and test qids disjoint ({'PASS' if g['disjoint'] else 'FAIL'}).",
         f"- Rows with a consensus label: {g['n_scored']} ({g['positives']} INCORRECT, {g['negatives']} CORRECT). "
         f"AMBIGUOUS rows excluded: {g['n_ambiguous_excluded']}."]
    pw = g["power_at_actual_n"]
    nr = g["n_required_for_power_0.80"]
    L.append(f"- Power to detect AUROC 0.65 versus 0.5 at the actual n: "
             f"{pw:.2f}" if pw is not None else "- Power: undefined (one class only).")
    if nr:
        L.append(f"- n required for 80 percent power at the observed prevalence: {nr['n']} "
                 f"({nr['pos']} positives, {nr['neg']} negatives). {'PASS' if g['n_scored'] >= nr['n'] else 'FAIL'}.")
    L += ["", "## Scorers", "", "| scorer | n | AUROC | vs length: diff, p | vs mean_logprob: diff, p | flag |",
          "|---|---:|---|---|---|---|"]
    for sc, r in g["scorers"].items():
        vl = r.get("vs_length")
        vm = r.get("vs_mean_logprob")
        fl = "control" if sc == "length" else ("" if r["beats_length"] else "not distinguishable from length")
        vl_s = f"{stats.f(vl['diff'])}, p={vl['p_delong']:.3f}" if vl else "n/a"
        vm_s = f"{stats.f(vm['diff'])}, p={vm['p_delong']:.3f}" if vm else "n/a"
        L.append(f"| {sc} | {r['n']} | {stats.f(r['auroc'])} | {vl_s} | {vm_s} | {fl} |")
    L += ["", "A scorer is flagged when its AUROC does not beat `length` at p < 0.05 (DeLong). "
          "Flagged scorers carry a marker in every later figure.", ""]
    return "\n".join(L)
