"""results/phase1_notes.md: per split and model, row count, judge agreement, kappa, AMBIGUOUS count, cost so far."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import cost, data, stats


def _agreement(lab: pd.DataFrame, model: str) -> dict:
    w = lab[lab["model"] == model].pivot(index="qid", columns="judge", values="verdict").dropna()
    a, b = w["judge_a"].to_numpy(), w["judge_b"].to_numpy()
    n = len(w)
    agree = int((a == b).sum())
    kappa = stats.boot(lambda x, y: float(cohen_kappa_score(x, y)) if len(np.unique(np.r_[x, y])) > 1 else None, a, b)
    cons = data.consensus(lab[lab["model"] == model])
    return {"n": n, "agreement": stats.wilson(agree, n), "kappa": kappa,
            "ambiguous": int((cons["label"] == "AMBIGUOUS").sum()),
            "incorrect": int((cons["label"] == "INCORRECT").sum()),
            "judge_a_incorrect": int((a == "INCORRECT").sum()), "judge_b_incorrect": int((b == "INCORRECT").sum())}


def write_phase1_notes(extra: list | None = None) -> str:
    calls = cost.summary()
    L = ["# Phase 1 notes: data, generations, labels", "",
         "Agreement carries a Wilson interval; kappa a bootstrap interval (2000 resamples, seed 42). Counts are exact.", ""]
    for split in data.SPLITS:
        d = data.load_split(split)
        g = d["generations"]
        L += [f"## {split}", "", f"- Questions: {len(d['questions'])}. Generation rows: {len(g)}.",
              "- Rows per role: " + ", ".join(f"{r}={int((g['role'] == r).sum())}" for r in data.ROLES) + "."]
        for model in sorted(d["labels"]["model"].unique()):
            a = _agreement(d["labels"], model)
            L.append(f"- `{model}`: {a['n']} judged rows, agreement {100 * a['agreement']['est']:.1f}% "
                     f"[{100 * a['agreement']['lo']:.1f}, {100 * a['agreement']['hi']:.1f}], kappa {stats.f(a['kappa'])}, "
                     f"consensus INCORRECT {a['incorrect']}, AMBIGUOUS (excluded) {a['ambiguous']}. "
                     f"Judge A marked {a['judge_a_incorrect']} INCORRECT, judge B {a['judge_b_incorrect']}.")
        L.append("")
    L += ["## Cost so far", "", f"From results/calls.jsonl, priced on {calls['priced_on']}: {calls['n_calls']} calls, "
          f"{calls['input_tokens']} input tokens, {calls['output_tokens']} output tokens, ${calls['cost_usd']:.4f}.", ""]
    for k in sorted(calls["by_purpose"]):
        b = calls["by_purpose"][k]
        L.append(f"- {k}: {b['n_calls']} calls, ${b['cost_usd']:.4f}")
    cfg = data.models_config()
    L += ["", "## Limitations", "",
          f"- Two judges from one provider: judge_a is `{cfg['judge_a']}` and judge_b is `{cfg['judge_b']}`, both OpenAI. "
          "Two judges from one provider share more failure modes than two providers would, so agreement overstates label reliability.",
          "- All 162 calibration primary rows were regenerated: the v1 CSVs store only a scalar mean logprob, not per-token logprobs.",
          "- Labels are from LLM judges, not humans."]
    if extra:
        L += ["", "## Notes", ""] + [f"- {e}" for e in extra]
    L.append("")
    text = "\n".join(L)
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    (data.RESULTS / "phase1_notes.md").write_text(text)
    return text
