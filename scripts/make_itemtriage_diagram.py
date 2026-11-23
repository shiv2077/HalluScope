"""Generate docs/itemtriage_architecture.excalidraw. Run: python3 scripts/make_itemtriage_diagram.py"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "itemtriage_architecture.excalidraw"
els: list[dict] = []
_n = [0]


def _id(p: str) -> str:
    _n[0] += 1
    return f"{p}{_n[0]}"


def _base(kind, x, y, w, h, **kw):
    return {"id": _id(kind[0]), "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
            "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
            "roughness": 1, "opacity": 100, "seed": _n[0], "version": 1, "versionNonce": _n[0], "isDeleted": False,
            "groupIds": [], "boundElements": [], "updated": 1, "link": None, "locked": False, "frameId": None, **kw}


def box(x, y, w, h, text, color="#ffffff", fs=14):
    r = _base("rectangle", x, y, w, h, backgroundColor=color, roundness={"type": 3})
    t = _base("text", x + 8, y + 8, w - 16, h - 16, strokeWidth=1, text=text, fontSize=fs, fontFamily=1,
              textAlign="center", verticalAlign="middle", containerId=r["id"], originalText=text, lineHeight=1.25,
              baseline=fs)
    r["boundElements"] = [{"id": t["id"], "type": "text"}]
    els.extend([r, t])
    return r["id"]


def arrow(x1, y1, x2, y2, a=None, b=None):
    els.append(_base("arrow", x1, y1, x2 - x1, y2 - y1, points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None,
                     startArrowhead=None, endArrowhead="arrow", roundness={"type": 2},
                     startBinding={"elementId": a, "focus": 0, "gap": 4} if a else None,
                     endBinding={"elementId": b, "focus": 0, "gap": 4} if b else None))


ingest = box(30, 230, 190, 100, "MMLU-Redux 2.0\nitems table, labels apart\nsubject split, sha256", "#e9ecef")
gsm = box(30, 380, 190, 80, "platinum GSM8K\nblind replication (H4)", "#e9ecef", fs=12)
loc = box(280, 60, 230, 90, "own panel: 5 local models\nzero-shot letters, two orders\nmeasured GPU cost", "#fff3bf", fs=12)
api = box(280, 180, 230, 80, "API panel (gpt-4o, mini)\nnot run: key invalid", "#f1f3f5", fs=12)
MX_TEXT = "public matrix: 395 leaderboard\nmodels, 5-shot log-likelihood\njoined by item text, 99.8%\n335 after the 95% agreement filter"
mx = box(280, 290, 230, 110, MX_TEXT, "#fff3bf", fs=12)
smp = box(570, 230, 210, 110, "panel sampler\nk = 1 to all\ntop, bottom, random,\nlineage-diverse", "#d3f9d8", fs=12)
SC_TEXT = "label-free scorers\ndisagree, strong3, item_total,\nirt_small, conf, conf_w,\nsubject_mean, within_subject,\nlength (control)"
sc = box(840, 200, 230, 170, SC_TEXT, "#d3f9d8", fs=12)
rk = box(1130, 230, 190, 110, "rank under a review budget\nties in expectation", "#d0ebff", fs=12)
lab = box(840, 440, 230, 90, "labels loader\nallowlisted callers only", "#e5dbff", fs=12)
ev = box(1130, 420, 190, 130, "evaluate: precision at\n5, 10, 20% budgets\nnested bootstrap,\nPREREG commit checked", "#e5dbff", fs=12)
out = box(1380, 250, 170, 90, "expert review\ntop of the ranking", "#ffc9c9")
arrow(220, 280, 280, 105, ingest, loc)
arrow(220, 280, 280, 220, ingest, api)
arrow(220, 280, 280, 345, ingest, mx)
arrow(220, 420, 280, 370, gsm, mx)
arrow(510, 105, 570, 260, loc, smp)
arrow(510, 345, 570, 300, mx, smp)
arrow(780, 285, 840, 285, smp, sc)
arrow(1070, 285, 1130, 285, sc, rk)
arrow(1320, 285, 1380, 295, rk, out)
arrow(125, 330, 840, 485, ingest, lab)
arrow(1070, 485, 1130, 485, lab, ev)
arrow(1225, 340, 1225, 420, rk, ev)

OUT.write_text(json.dumps({"type": "excalidraw", "version": 2, "source": "scripts/make_itemtriage_diagram.py",
                           "elements": els, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}},
                          indent=1))
print(f"wrote {OUT} ({len(els)} elements)")
