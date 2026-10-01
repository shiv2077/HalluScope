"""Pre-flight audit: render every prompt the panel will send and run the full leak guard against all
annotation text. Allowed to read labels; the panel runners are not. Writes results/itemtriage/prompt_audit.json.
"""

from __future__ import annotations

import json

import pandas as pd

from . import labels as labels_mod
from . import paths, prompts

LOCAL_SHIFTS = (0, 1)


def run() -> dict:
    items = pd.read_parquet(paths.ITEMS)
    lab = labels_mod.load(caller="itemtriage.audit")
    reasons = tuple(t for t in lab["potential_reason"] if len(str(t).strip()) >= 20)
    corrects = tuple(t for t in lab["correct_answer_raw"] if len(str(t).strip()) >= 20)
    forbidden = reasons + corrects
    n = 0
    for it in items.to_dict("records"):
        for shift in LOCAL_SHIFTS:
            text, order = prompts.mc_prompt(it, shift)
            prompts.check_prompt(text, it, order, forbidden)
            n += 1
        prompts.check_prompt(prompts.ptrue_prompt(it), it, None, forbidden)
        n += 1
    out = {"items": len(items), "prompts_checked": n, "forbidden_texts": len(forbidden), "violations": 0}
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    (paths.RESULTS / "prompt_audit.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return out
