"""Verification script for Phase 2 Task 2.1+2.2.

Runs a tiny local-mode generation and asserts that StandardizedResult
now carries hidden_states_per_token and logits_per_token with the right
shapes. Uses max_tokens=5 to keep wall-clock low; the slow part is the
one-time model download (~2.2GB on first run).

Run from repo root:
    python scripts/test_hidden_states.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.adapters.local_adapter import LocalAdapter

print("Loading LocalAdapter (this triggers tokenizer + model load)...")
adapter = LocalAdapter()

print("Generating 5 tokens...")
result = adapter.generate(
    prompt="The capital of France is",
    max_tokens=5,
    temperature=0.0,  # deterministic for reproducible shapes
)

print()
print("=== StandardizedResult basics ===")
print("  text                    :", repr(result.text))
print("  model_name              :", result.model_name)
print("  len(token_logprobs)     :", len(result.token_logprobs))
print("  len(token_strings)      :", len(result.token_strings))

print()
print("=== Phase 2 internal-signal fields ===")
hsp = result.hidden_states_per_token
lpt = result.logits_per_token

assert hsp is not None, "hidden_states_per_token must not be None for local adapter"
assert lpt is not None, "logits_per_token must not be None for local adapter"

print("  hidden_states_per_token : list of", len(hsp), "elements")
print("  hidden_states[0].shape  :", hsp[0].shape, "dtype:", hsp[0].dtype)
print("  logits_per_token        : list of", len(lpt), "elements")
print("  logits[0].shape         :", lpt[0].shape, "dtype:", lpt[0].dtype)

# ── Shape assertions ────────────────────────────────────────────────
assert len(hsp) == len(
    result.token_logprobs
), f"hidden_states list length ({len(hsp)}) != logprobs ({len(result.token_logprobs)})"
assert len(lpt) == len(
    result.token_logprobs
), f"logits list length ({len(lpt)}) != logprobs ({len(result.token_logprobs)})"

# TinyLlama: hidden_dim=2048, vocab=32000. Don't hard-code — surface actual.
hidden_dim = hsp[0].shape[0]
vocab_size = lpt[0].shape[0]
print()
print(f"  detected hidden_dim     : {hidden_dim}")
print(f"  detected vocab_size     : {vocab_size}")

assert hsp[0].ndim == 1, f"hidden state should be 1-D vector, got shape {hsp[0].shape}"
assert lpt[0].ndim == 1, f"logits should be 1-D vector, got shape {lpt[0].shape}"

# Sanity: vectors should be finite
import numpy as np

assert np.all(np.isfinite(hsp[0])), "hidden state contains nan/inf"
assert np.all(np.isfinite(lpt[0])), "logits contain nan/inf"

print()
print("ALL ASSERTIONS PASSED.")
