"""The only reader of data/itemtriage/labels.parquet. Callers must be on the allowlist.

Scorers, prompts and panel runners are not on it, so a label can never reach a score or a prompt.
"""

from __future__ import annotations

import inspect

import pandas as pd

from . import paths

ALLOWED_CALLERS = ("itemtriage.splits", "itemtriage.evaluate", "itemtriage.calibrate", "itemtriage.supervised",
                   "itemtriage.gates", "itemtriage.audit", "itemtriage.human", "itemtriage.findings")


def load(caller: str | None = None) -> pd.DataFrame:
    mod = caller or inspect.getmodule(inspect.stack()[1].frame).__name__
    if not (mod in ALLOWED_CALLERS or mod.startswith("tests.") or mod.startswith("test_")):
        raise PermissionError(f"{mod} may not read labels; allowed: {ALLOWED_CALLERS}")
    return pd.read_parquet(paths.LABELS)
