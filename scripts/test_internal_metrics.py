"""Verification for Task 2.3 — Semantic Energy + RCD on synthetic vectors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np

from backend.app.metrics.internal_metrics import (
    semantic_energy,
    representation_collapse,
    aggregate_internal_signals,
)

rng = np.random.default_rng(42)

# ── Random sanity ───────────────────────────────────────────────────
random_logits = [rng.normal(0, 5, size=32000).astype("float32") for _ in range(5)]
random_hidden = [rng.normal(0, 1, size=2048).astype("float32") for _ in range(5)]

E = semantic_energy(random_logits)
R = representation_collapse(random_hidden)

print("--- Random synthetic ---")
print("  semantic_energy        :", [round(e, 3) for e in E])
print("  representation_collapse:", [round(r, 4) for r in R])
print("  len(R)                 :", len(R), "(expected", len(random_hidden) - 1, ")")
assert len(R) == len(random_hidden) - 1
assert all(np.isfinite(e) for e in E), "E contains nan/inf"
assert all(np.isfinite(r) for r in R), "R contains nan/inf"

# ── RCD edge cases ──────────────────────────────────────────────────
print()
print("--- RCD edge cases ---")
v = rng.normal(0, 1, size=2048).astype("float32")

# identical vectors -> cos=1 -> RCD=0
identical_pair = [v, v.copy()]
r_id = representation_collapse(identical_pair)
print(f"  identical    : {r_id[0]:.2e}  (expected ~0)")
assert abs(r_id[0]) < 1e-5

# orthogonal vectors -> cos=0 -> RCD=1
e1 = np.zeros(2048, dtype="float32")
e1[0] = 1.0
e2 = np.zeros(2048, dtype="float32")
e2[1] = 1.0
r_orth = representation_collapse([e1, e2])
print(f"  orthogonal   : {r_orth[0]:.6f}  (expected ~1)")
assert abs(r_orth[0] - 1.0) < 1e-5

# anti-parallel vectors -> cos=-1 -> RCD=2
r_anti = representation_collapse([e1, -e1])
print(f"  anti-parallel: {r_anti[0]:.6f}  (expected ~2)")
assert abs(r_anti[0] - 2.0) < 1e-5

# single hidden state -> empty RCD list
r_one = representation_collapse([e1])
assert r_one == [], f"expected [], got {r_one}"
print("  single token : []  (expected [])")

# ── Semantic energy sanity ─────────────────────────────────────────
print()
print("--- Semantic energy edge cases ---")

# Uniform logits over a smaller vocab -> E = -log(K). For K=10 with all zeros:
uniform = [np.zeros(10, dtype="float32")]
E_uniform = semantic_energy(uniform)
expected_uniform = -math.log(10.0)
print(f"  uniform K=10 : E={E_uniform[0]:.6f}  expected={expected_uniform:.6f}")
assert abs(E_uniform[0] - expected_uniform) < 1e-5

# Spiky distribution (one huge logit) -> E ≈ -that_logit
spiky = np.zeros(1000, dtype="float32")
spiky[0] = 50.0
E_spiky = semantic_energy([spiky])
print(f"  spiky (max=50): E={E_spiky[0]:.4f}  (expected ~-50)")
assert abs(E_spiky[0] - (-50.0)) < 1e-3

# Numerical stability: very large logits should not overflow
huge = np.full(32000, 700.0, dtype="float32")
E_huge = semantic_energy([huge])
print(f"  all 700      : E={E_huge[0]:.4f}  (must be finite)")
assert np.isfinite(E_huge[0])

# ── Aggregate ──────────────────────────────────────────────────────
print()
print("--- aggregate_internal_signals ---")
agg = aggregate_internal_signals(random_logits, random_hidden)
for k, v in agg.items():
    if isinstance(v, list):
        print(f"  {k:>22}: list[{len(v)}] = {v[:5]}{'...' if len(v) > 5 else ''}")
    else:
        print(f"  {k:>22}: {v}")
assert agg["mean_semantic_energy"] is not None
assert agg["mean_rcd"] is not None
assert agg["min_rcd"] is not None
assert len(agg["per_token_energy"]) == 5
assert len(agg["per_token_rcd"]) == 4

# API-mode short-circuit
agg_none = aggregate_internal_signals(None, None)
assert agg_none["mean_semantic_energy"] is None
assert agg_none["mean_rcd"] is None
print()
print("  API-mode short-circuit returns Nones: OK")

print()
print("ALL ASSERTIONS PASSED.")
