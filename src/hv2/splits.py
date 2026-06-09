"""Question-disjoint calib/val/test splits — created once, hashed, frozen.

CLAUDE.md HARD RULE: splits are disjoint BY QUESTION, created once in E1, hashed,
and never modified. Assignment is a deterministic function of the qid hash (not
an RNG over order), so it is identical across machines and re-runs and is robust
to the question pool growing — an existing qid keeps its split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLITS = ("calib", "val", "test")


def assign_split(qid: str, fractions: tuple[float, float, float] = (0.4, 0.3, 0.3)) -> str:
    """Deterministically map a qid to calib/val/test via its hash.

    Stable: depends only on the qid string, not on insertion order or n.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1"
    h = int(hashlib.md5(qid.encode()).hexdigest(), 16) / 16**32  # in [0,1)
    c, v = fractions[0], fractions[0] + fractions[1]
    if h < c:
        return "calib"
    if h < v:
        return "val"
    return "test"


def make_splits(qids: list[str], fractions: tuple[float, float, float] = (0.4, 0.3, 0.3)) -> dict[str, str]:
    return {qid: assign_split(qid, fractions) for qid in qids}


def splits_payload(qids: list[str], fractions: tuple[float, float, float] = (0.4, 0.3, 0.3)) -> dict:
    mapping = make_splits(qids, fractions)
    counts = {s: sum(1 for v in mapping.values() if v == s) for s in SPLITS}
    # content hash over the sorted (qid,split) pairs -> detects any drift
    blob = json.dumps(sorted(mapping.items()), sort_keys=True).encode()
    return {
        "fractions": list(fractions),
        "counts": counts,
        "n": len(qids),
        "content_hash": hashlib.sha256(blob).hexdigest(),
        "mapping": mapping,
    }


def freeze_splits(qids: list[str], path: str | Path, fractions=(0.4, 0.3, 0.3)) -> dict:
    """Write splits once. If the file exists, VERIFY it matches and refuse to
    overwrite (HARD RULE: splits never modified)."""
    path = Path(path)
    payload = splits_payload(qids, fractions)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"Refusing to modify frozen splits at {path}: content hash "
                f"changed ({existing.get('content_hash')} -> {payload['content_hash']}). "
                "Splits are immutable per CLAUDE.md. Delete intentionally if you "
                "truly mean to recreate them."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload
