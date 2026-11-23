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


ingest = box(30, 210, 180, 90, "MMLU-Redux 2.0\ningest: items table\n+ separate labels table", "#e9ecef")
split = box(30, 360, 180, 90, "subject split\nstratified by base rate\nfrozen with sha256", "#e9ecef")
guard = box(260, 60, 210, 90, "prompt guard\noptions verbatim, no category ids,\nno annotation text", "#fff0f6", fs=12)
api = box(260, 180, 210, 80, "API panel: gpt-4o, gpt-4o-mini\nletter logprobs + P(True)", "#fff3bf", fs=12)
loc = box(260, 290, 210, 90, "local panel on 6 GB GPU\nQwen 3B/1.5B, Phi-3.5, SmolLM2, Granite\ntwo option orders", "#fff3bf", fs=12)
hid = box(260, 410, 210, 80, "hidden states\nQwen2.5-1.5B, key + last prompt token", "#fff3bf", fs=12)
SCORERS_TEXT = "label-free scorers\ndisagree, conf, conf_w (IRT-weighted)\nsmall-panel 4PL forced ceiling\nptrue_key, length, dup_options"
sc = box(530, 230, 210, 120, SCORERS_TEXT, "#d3f9d8", fs=12)
rk = box(800, 230, 190, 120, "rank under a review budget\ndeclared tie key,\nties in expectation", "#d0ebff", fs=12)
lab = box(530, 420, 210, 90, "labels loader\nallowlisted callers only", "#e5dbff", fs=12)
ev = box(800, 420, 190, 90, "evaluate on test\nonly if PREREG is committed\nunblinding recorded", "#e5dbff", fs=12)
out = box(1050, 300, 170, 110, "expert review\ntop of the ranking", "#ffc9c9")
arrow(120, 300, 120, 360, ingest, split)
arrow(210, 255, 260, 220, ingest, api)
arrow(210, 255, 260, 335, ingest, loc)
arrow(365, 150, 365, 180, guard, api)
arrow(470, 220, 530, 270, api, sc)
arrow(470, 335, 530, 300, loc, sc)
arrow(470, 450, 530, 330, hid, sc)
arrow(740, 290, 800, 290, sc, rk)
arrow(990, 290, 1050, 340, rk, out)
arrow(210, 405, 530, 465, split, lab)
arrow(740, 465, 800, 465, lab, ev)
arrow(895, 350, 895, 420, rk, ev)

OUT.write_text(json.dumps({"type": "excalidraw", "version": 2, "source": "scripts/make_itemtriage_diagram.py",
                           "elements": els, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}},
                          indent=1))
print(f"wrote {OUT} ({len(els)} elements)")
