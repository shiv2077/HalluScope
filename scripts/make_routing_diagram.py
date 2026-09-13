"""Generate docs/routing_architecture.excalidraw. Run: python3 scripts/make_routing_diagram.py"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "routing_architecture.excalidraw"
_id = 0
els = []


def nid(p):
    global _id
    _id += 1
    return f"{p}{_id}"


def box(x, y, w, h, text, color="#ffffff", stroke="#1e1e1e", fs=16):
    i = nid("b")
    t = nid("t")
    els.append({"id": i, "type": "rectangle", "x": x, "y": y, "width": w, "height": h, "angle": 0,
                "strokeColor": stroke, "backgroundColor": color, "fillStyle": "solid", "strokeWidth": 2,
                "roughness": 1, "opacity": 100, "roundness": {"type": 3}, "seed": _id, "version": 1,
                "versionNonce": _id, "isDeleted": False, "groupIds": [], "boundElements": [{"id": t, "type": "text"}],
                "updated": 1, "link": None, "locked": False, "frameId": None})
    els.append({"id": t, "type": "text", "x": x + 8, "y": y + 8, "width": w - 16, "height": h - 16, "angle": 0,
                "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 1,
                "roughness": 1, "opacity": 100, "seed": _id, "version": 1, "versionNonce": _id, "isDeleted": False,
                "groupIds": [], "boundElements": [], "updated": 1, "link": None, "locked": False, "frameId": None,
                "text": text, "fontSize": fs, "fontFamily": 1, "textAlign": "center", "verticalAlign": "middle",
                "containerId": i, "originalText": text, "lineHeight": 1.25, "baseline": fs})
    return i


def arrow(x1, y1, x2, y2, label=None, a=None, b=None):
    i = nid("a")
    el = {"id": i, "type": "arrow", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "angle": 0,
          "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
          "roughness": 1, "opacity": 100, "seed": _id, "version": 1, "versionNonce": _id, "isDeleted": False,
          "groupIds": [], "boundElements": [], "updated": 1, "link": None, "locked": False, "frameId": None,
          "points": [[0, 0], [x2 - x1, y2 - y1]], "lastCommittedPoint": None, "startArrowhead": None,
          "endArrowhead": "arrow", "roundness": {"type": 2},
          "startBinding": {"elementId": a, "focus": 0, "gap": 4} if a else None,
          "endBinding": {"elementId": b, "focus": 0, "gap": 4} if b else None}
    els.append(el)
    if label:
        t = nid("t")
        els.append({"id": t, "type": "text", "x": (x1 + x2) / 2 - 60, "y": (y1 + y2) / 2 - 24, "width": 120, "height": 20,
                    "angle": 0, "strokeColor": "#5c5c5c", "backgroundColor": "transparent", "fillStyle": "solid",
                    "strokeWidth": 1, "roughness": 1, "opacity": 100, "seed": _id, "version": 1, "versionNonce": _id,
                    "isDeleted": False, "groupIds": [], "boundElements": [], "updated": 1, "link": None, "locked": False,
                    "frameId": None, "text": label, "fontSize": 13, "fontFamily": 1, "textAlign": "center",
                    "verticalAlign": "top", "containerId": None, "originalText": label, "lineHeight": 1.25, "baseline": 13})
    return i


# main flow, left to right
inp = box(40, 200, 170, 80, "model output\n(answer + tokens)", "#e9ecef")
sel = box(270, 180, 220, 120, "scorer tier selection\nby provider capability\n\nT0 logprobs free\nT1 consistency, 3 calls\nT2 P(True), 1 call", "#fff3bf", fs=13)
sco = box(550, 200, 170, 80, "score\n+ cost per score", "#d3f9d8")
thr = box(780, 200, 170, 80, "threshold\n(from calibration.json)", "#d0ebff")
ret = box(1020, 80, 170, 70, "return", "#d3f9d8")
esc = box(1020, 205, 170, 70, "escalate\n(strong model)", "#ffe8cc")
rev = box(1020, 330, 170, 70, "review\n(human)", "#ffc9c9")
arrow(210, 240, 270, 240, a=inp, b=sel)
arrow(490, 240, 550, 240, a=sel, b=sco)
arrow(720, 240, 780, 240, a=sco, b=thr)
arrow(950, 230, 1020, 115, "low risk", a=thr, b=ret)
arrow(950, 240, 1020, 240, "mid risk", a=thr, b=esc)
arrow(950, 250, 1020, 365, "high risk", a=thr, b=rev)

# calibration feeding the threshold
lab = box(560, 400, 170, 80, "labelled sample\n(per deployment,\nblind two judges)", "#e5dbff", fs=13)
cal = box(780, 400, 170, 80, "calibrate()\nbudget + risk cutoffs\ncoverage, risk, n, hash", "#e5dbff", fs=13)
arrow(730, 440, 780, 440, a=lab, b=cal)
arrow(865, 400, 865, 280, a=cal, b=thr)

# validity gate beside the scorer
gate = box(270, 340, 220, 140, "validity gates\n\n1 template contamination\n2 split disjointness\n3 min n for AUROC power\n4 beats length (DeLong)", "#fff0f6", fs=13)
arrow(380, 340, 380, 300, a=gate, b=sel)

doc = {"type": "excalidraw", "version": 2, "source": "scripts/make_routing_diagram.py", "elements": els,
       "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}
OUT.write_text(json.dumps(doc, indent=1))
print(f"wrote {OUT} ({len(els)} elements)")
