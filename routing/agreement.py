"""Agreement between human labels and the machine labels, plus lift under a human oracle.

Writes results/human_agreement.md. Runs only when data/<split>_human_labels.csv exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import data, stats
from .label_tool import human_path
from .policy import flag_by_budget

BUDGET = 0.2
LABS = ["CORRECT", "INCORRECT", "AMBIGUOUS"]


def _kappa(a, b):
    return None if len(np.unique(np.r_[a, b])) < 2 else float(cohen_kappa_score(a, b))


def _pair(name: str, human: np.ndarray, other: np.ndarray) -> dict:
    n = len(human)
    return {"pair": name, "n": n, "agreement": stats.wilson(int((human == other).sum()), n),
            "kappa": stats.boot(_kappa, human, other),
            "confusion": pd.crosstab(pd.Series(human, name="human"), pd.Series(other, name=name)).reindex(index=LABS, columns=LABS, fill_value=0)}


def _lift_block(y: np.ndarray, s: np.ndarray) -> dict:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]

    def fl(ss):
        return flag_by_budget(pd.Series(ss), BUDGET).to_numpy()

    def catch(yy, ss):
        f = fl(ss)
        return None if yy.sum() == 0 or f.sum() == 0 else float(yy[f].sum() / yy.sum())

    def lift(yy, ss):
        f = fl(ss)
        return None if yy.sum() == 0 or f.sum() == 0 else float((yy[f].sum() / yy.sum()) / (f.sum() / len(f)))

    def prec(yy, ss):
        f = fl(ss)
        return None if f.sum() == 0 else float(yy[f].mean())

    return {"n": int(len(y)), "errors": int(y.sum()), "catch": stats.boot(catch, y, s), "lift": stats.boot(lift, y, s),
            "precision": stats.boot(prec, y, s)}


def run(split: str = "test") -> str | None:
    p = human_path(split)
    if not p.exists():
        print(f"no human labels at {p}; run `python3 -m routing.cli label --split {split} ...` first")
        return None
    d = data.load_split(split)
    cheap = data.models_config()["cheap"]
    h = pd.read_csv(p, dtype=str, keep_default_na=False)
    cons = d["consensus"]
    cons = cons[cons["model"] == cheap].set_index("qid")
    lab = d["labels"]
    lab = lab[lab["model"] == cheap].pivot(index="qid", columns="judge", values="verdict")
    L = ["# Human labels against machine labels", "",
         f"Split `{split}`, cheap model `{cheap}`. Agreement carries a Wilson interval, kappa a bootstrap interval "
         "(2000 resamples, seed 42). Kappa is over the three-way verdict CORRECT, INCORRECT, AMBIGUOUS.", ""]
    for annot, ha in h.groupby("annotator"):
        ha = ha.drop_duplicates("qid", keep="last").set_index("qid")
        qids = [q for q in ha.index if q in cons.index]
        hv = ha.loc[qids, "verdict"].to_numpy()
        pairs = [_pair("consensus", hv, cons.loc[qids, "label"].to_numpy()),
                 _pair("judge_a", hv, lab.loc[qids, "judge_a"].to_numpy()),
                 _pair("judge_b", hv, lab.loc[qids, "judge_b"].to_numpy())]
        L += [f"## Annotator `{annot}`: {len(qids)} labelled rows", "", "| pair | n | agreement | kappa |", "|---|---:|---|---|"]
        for pr in pairs:
            L.append(f"| human vs {pr['pair']} | {pr['n']} | {_pct(pr['agreement'])} | {stats.f(pr['kappa'])} |")
        for pr in pairs:
            L += ["", f"Confusion, rows human, columns {pr['pair']}:", "", "| human \\ " + pr["pair"] + " | " + " | ".join(LABS) + " |", "|---|" + "---:|" * 3]
            for r in LABS:
                L.append(f"| {r} | " + " | ".join(str(int(pr["confusion"].loc[r, c])) for c in LABS) + " |")
        dis = [q for q in qids if hv[qids.index(q)] != cons.loc[q, "label"]]
        L += ["", f"### Disagreements with consensus: {len(dis)}", ""]
        if dis:
            L += ["| qid | human | judge_a | judge_b | consensus | note |", "|---|---|---|---|---|---|"]
            for q in dis:
                L.append(f"| {q} | {ha.loc[q, 'verdict']} | {lab.loc[q, 'judge_a']} | {lab.loc[q, 'judge_b']} | {cons.loc[q, 'label']} | {ha.loc[q, 'note']} |")
        # lift under each oracle on the same rows, binary rows only
        sc = data.read_scores(split).set_index("qid")
        rows = [q for q in qids if ha.loc[q, "verdict"] != "AMBIGUOUS" and cons.loc[q, "label"] != "AMBIGUOUS"]
        yh = (ha.loc[rows, "verdict"] == "INCORRECT").astype(int).to_numpy()
        yc = (cons.loc[rows, "label"] == "INCORRECT").astype(int).to_numpy()
        L += ["", f"### Lift at the {BUDGET:.2f} budget on the {len(rows)} rows both oracles label CORRECT or INCORRECT "
              f"(human errors {int(yh.sum())}, consensus errors {int(yc.sum())})", "",
              "| scorer | oracle | catch rate | lift | precision |", "|---|---|---|---|---|"]
        for s_name in data.SCORERS:
            s = sc.loc[rows, s_name].astype(float).to_numpy()
            for oname, y in (("human", yh), ("consensus", yc)):
                b = _lift_block(y, s)
                L.append(f"| {s_name} | {oname} | {_pct(b['catch'])} | {stats.f(b['lift'], 2)} | {_pct(b['precision'])} |")
        L.append("")
    text = "\n".join(L)
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    (data.RESULTS / "human_agreement.md").write_text(text)
    print(f"wrote {data.RESULTS / 'human_agreement.md'}")
    return text


def _pct(d):
    return "n/a" if d is None else f"{100 * d['est']:.1f}% [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
