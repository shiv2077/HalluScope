# halluscope-core

The HRI scoring engine factored out of the [HalluScope](../../README.md) reference app: the
weighted-sigmoid aggregator (`risk/hri_engine.py`), the per-token uncertainty signals
(`metrics/`), and the config loader (`config_store.py`). No FastAPI, no SQLite, no React — this
package can be installed and imported on its own.

It is part of the HalluScope monorepo, not a separately versioned/released project (see
`docs/DECISIONS.md` in the repo root). The FastAPI app imports this package's public functions
through thin backward-compatible shims at their old `backend.app.*` paths, so application code
did not need to change when this package was extracted.

## Install

```bash
pip install -e packages/halluscope-core
```

## Usage

```python
from halluscope_core.risk.hri_engine import compute_hri

result = compute_hri(logprobs=[-0.05, -0.12, -0.08, -1.40, -0.30])
print(result["hri"], result["risk_level"])
# 0.0182 LOW
```

## Configuration

Reads `config/*.json` from the monorepo root by default (weights, thresholds, normalization
constants — see the repo root README's "HRI formula" section). Set `HALLUSCOPE_CONFIG_DIR` to
point at a different directory with the same JSON schema when using this package outside the
monorepo.

## What's not here

`semantic_novelty` (Stage 3, needs `sentence-transformers`) and the spaCy entity-extraction
backend are lazy-imported inside `metrics/semantic_checks.py` and `metrics/nlp_backend.py` and
are not required by `compute_hri` or the shipped HRI weights. Install the `nlp` extra
(`pip install -e "packages/halluscope-core[nlp]"`) only if you call those functions directly.
