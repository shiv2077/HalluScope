"""Phase 6 non-expert spot check: blind human labels on 150 test items from subjects the owner can judge.

Sample: the top 75 by conf_w (ties by its declared key) within the judgeable test subjects, plus 75 random
items from the rest of those subjects, shuffled together with seed 42. The terminal shows question, options
and keyed answer only; never a score, a pool, or a category. Verdicts append to
data/itemtriage/human_labels.csv and the tool resumes. The agreement report compares with the dataset labels.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from routing import stats

from . import paths, prompts, scorers, splits

JUDGEABLE = ("college_computer_science", "college_mathematics", "college_physics", "computer_security",
             "conceptual_physics", "elementary_mathematics", "high_school_mathematics", "high_school_physics",
             "machine_learning")
N_TOP, N_RANDOM, SEED = 75, 75, 42
SAMPLE = paths.DATA / "human_sample.json"
LABELS_CSV = paths.DATA / "human_labels.csv"
COLS = ["item_id", "annotator", "verdict", "note", "timestamp"]
KEYS = {"f": "flawed", "o": "ok", "u": "unsure"}
SCORE_WORDS = tuple(scorers.LABEL_FREE) + ("score", "pool", "flagged", "random")


def sample() -> list[str]:
    if SAMPLE.exists():
        return json.loads(SAMPLE.read_text())["item_ids"]
    sc = pd.read_parquet(scorers.SCORES)
    test = set(splits.load()["item_ids"]["test"])
    pool = sc[sc["item_id"].isin(test) & sc["subject"].isin(JUDGEABLE)].sort_values("item_id")
    order = np.lexsort((-pool[scorers.TIE_KEY["conf_w"]].to_numpy(), -pool["conf_w"].to_numpy()))
    top = pool.iloc[order[:N_TOP]]["item_id"].tolist()
    rest = sorted(set(pool["item_id"]) - set(top))
    rng = np.random.RandomState(SEED)
    rand = rng.choice(rest, N_RANDOM, replace=False).tolist()
    ids = list(rng.permutation(top + rand))
    SAMPLE.write_text(json.dumps({"item_ids": ids, "top": sorted(top), "random": sorted(rand), "seed": SEED}, indent=1))
    return ids


def render(item: dict) -> str:
    opts = list(item["options"])
    k = int(item["key_index"])
    text = (f"\nSubject: {prompts.subject_name(item['subject'])}\n\n{item['question']}\n"
            + "\n".join(f"{paths.LETTERS[i]}. {o}" for i, o in enumerate(opts))
            + f"\n\nAnswer key: {paths.LETTERS[k]}. {opts[k]}\n"
            "Is this item sound? [o = ok, f = flawed (wrong key, no correct option, or several correct), u = unsure], "
            "optional note after a space: ")
    low = text.lower()
    assert not prompts.CATEGORY_RE.search(text), "display leaks a category identifier"
    own = (item["question"] + " " + " ".join(opts)).lower()
    leaked = [w for w in SCORE_WORDS if w in low and w not in own]
    assert not leaked, f"display leaks {leaked}"
    return text


def run(annotator: str, stdin=None, stdout=None) -> int:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    ids = sample()
    items = pd.read_parquet(paths.ITEMS).set_index("item_id")
    done = set()
    if LABELS_CSV.exists():
        d = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
        done = set(d.loc[d["annotator"] == annotator, "item_id"])
    n = 0
    for i, iid in enumerate([x for x in ids if x not in done], len(done) + 1):
        stdout.write(f"[{i}/{len(ids)}]" + render(items.loc[iid].to_dict() | {"item_id": iid}))
        stdout.flush()
        line = stdin.readline()
        if not line:
            break
        key, _, note = line.strip().partition(" ")
        while key.lower() not in KEYS:
            stdout.write("Type o, f or u: ")
            stdout.flush()
            line = stdin.readline()
            if not line:
                return n
            key, _, note = line.strip().partition(" ")
        new = not LABELS_CSV.exists()
        with open(LABELS_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow({"item_id": iid, "annotator": annotator, "verdict": KEYS[key.lower()], "note": note.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        n += 1
    stdout.write(f"\n{n} labelled this session.\n")
    return n


def agreement() -> dict | None:
    """Precision on the human labels versus the dataset labels, per pool. Reads labels (allowlisted caller)."""
    from . import labels as labels_mod

    if not LABELS_CSV.exists():
        return None
    s = json.loads(SAMPLE.read_text())
    h = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False).drop_duplicates(["item_id", "annotator"], keep="last")
    lab = labels_mod.load(caller="itemtriage.human").set_index("item_id")
    out = {}
    for annot, g in h.groupby("annotator"):
        g = g.set_index("item_id")
        res = {}
        for pool in ("top", "random"):
            ids = [i for i in s[pool] if i in g.index]
            hv = g.loc[ids, "verdict"]
            decided = hv[hv != "unsure"]
            ds = lab.loc[ids, "flaw_primary"]
            res[pool] = {"labelled": len(ids), "unsure": int((hv == "unsure").sum()),
                         "human_flawed_rate": stats.wilson(int((decided == "flawed").sum()), len(decided)),
                         "dataset_flawed_rate": stats.wilson(int(ds.sum()), int(ds.notna().sum()))}
        both = [i for i in g.index if g.loc[i, "verdict"] != "unsure" and pd.notna(lab.loc[i, "flaw_primary"])]
        hb = (g.loc[both, "verdict"] == "flawed").to_numpy(int)
        db = lab.loc[both, "flaw_primary"].to_numpy(int)
        from sklearn.metrics import cohen_kappa_score

        res["agreement_with_dataset"] = stats.wilson(int((hb == db).sum()), len(both))
        res["kappa"] = stats.boot(lambda a, b: float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None, hb, db)
        out[annot] = res
    return out
