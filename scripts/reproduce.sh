#!/usr/bin/env bash
# HalluScope — reproduce the HaluEval-QA benchmark end to end.
#
# Mirrors the "Reproduce" section of eval/REPORT.md exactly; this script
# just chains those commands with expected runtime / cost noted inline.
# NOT executed as part of this revamp — review before running.
#
# Requirements:
#   - Repo venv active with backend/requirements.txt + eval deps installed
#     (scikit-learn, sentence-transformers, matplotlib, datasets — see
#     README.md "Tech stack"; not currently pinned in a single requirements
#     file; the routing layer has its own pinned pyproject.toml).
#   - OPENAI_API_KEY set in .env for the API-mode run.
#   - eval/run_benchmark.py drives the FastAPI app in-process via
#     starlette's TestClient — no separate `uvicorn` process needed.
#
# Total expected cost: ~$0.02 in OpenAI API calls, ~10-12 minutes wall time
# on a single machine (local-mode timing assumes a GPU; CPU-only local runs
# take substantially longer per REPORT.md's own caveats).

set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Build the 100-question calibration subset (seed=42, deterministic).
#    Runtime: seconds. API calls: 0 (pulls from the HuggingFace datasets cache).
python3 eval/build_subset.py

# 2. Run the benchmark in API mode.
#    Runtime: ~5 min. OpenAI API calls: 100 (one gpt-4o-mini chat completion
#    per calibration question). Cost: ~$0.02 (gpt-4o-mini pricing, ~80 max
#    tokens/answer). Add --eval-deterministic for a reproducible run
#    (greedy decoding, fixed seed) — REPORT.md's numbers used it.
python3 eval/run_benchmark.py --mode api --eval-deterministic

# 3. Run the benchmark in local mode.
#    Runtime: ~1.5 min on GPU (REPORT.md's figure); CPU-only will be much
#    slower — TinyLlama-1.1B-Chat, 100 questions, max_new_tokens<=80.
#    OpenAI API calls: 0 (fully local).
python3 eval/run_benchmark.py --mode local --eval-deterministic

# 4. Grade both runs: SBERT (all-MiniLM-L6-v2) cosine similarity of the
#    model's free-form answer against the dataset reference, threshold 0.35.
#    Runtime: seconds (one local SBERT forward pass per row). API calls: 0.
python3 eval/match.py --csv eval/results_api.csv   --threshold 0.35
python3 eval/match.py --csv eval/results_local.csv --threshold 0.35

# 5. Compute AUROC, best-F1 threshold, and ROC plots for both modes.
#    Runtime: seconds. API calls: 0. Writes eval/summary.json + eval/roc_*.png.
python3 eval/analyze.py

# 6. Sweep thresholds under the precision-floor constraints and select the
#    operating point (writes eval/operating_point.json). Runtime: seconds.
python3 eval/pick_thresholds.py

echo "Done. See eval/REPORT.md for how to read the outputs."
