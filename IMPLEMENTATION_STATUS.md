# HalluScope — Implementation Status (branch `feature/full-pipeline`)

Self-check after building the system the paper describes. **No experimental
numbers appear here or were produced.** Code builds *capability*; a human runs
the experiments (commands in §3) and records the real outputs. This file is not
the manuscript and contains no results tables.

73 unit tests pass (`python3 -m pytest tests/`). Every new signal has a
direction test proving higher value ⇒ higher risk (or documenting why not).

---

## 1. Paper-claim → code status

### Correctness fixes (were broken)
| Item | Status | Where |
|---|---|---|
| Semantic energy direction | **FIXED** | [internal_metrics.py](backend/app/metrics/internal_metrics.py)·`semantic_energy` — bounded flatness `F=(logsumexp−max)/logV ∈[0,1]`, shift-invariant; was inverted `abs(−logsumexp)/35`. Raw `free_energy()` kept separately. |
| Single HIGH threshold | **FIXED** | [config/thresholds.json](config/thresholds.json) read by `classify_risk`, `should_intervene`, `analyze.py`. 0.30/0.46/0.086 drift gone. |
| Weights/constants from config | **DONE** | [config/](config/) + [config_store.py](backend/app/config_store.py). No numeric risk literals remain in `hri_engine.py`. |
| Seeded determinism | **DONE** | [seeding.py](backend/app/seeding.py); `run_benchmark --eval-deterministic` (greedy+seed). |
| SBERT grader threshold explicit + sensitivity | **DONE** | [config/grader.json](config/grader.json) (default 0.60, the spec value) + [grader_sensitivity.py](eval/grader_sensitivity.py). |

### Stage 1 — pre-generation prompt risk ([prompt_risk.py](backend/app/metrics/prompt_risk.py))
| Signal | Status | Function |
|---|---|---|
| Entity rarity (s1) | **IMPLEMENTED** | `entity_rarity_signal` (via [nlp_backend.py](backend/app/metrics/nlp_backend.py)`.rare_entity_fraction`) |
| Time risk (s2) | **IMPLEMENTED** | `time_risk` (post-cutoff years + relative markers) |
| Prompt perplexity (s3) | **IMPLEMENTED** (local); **PARTIAL** (API) | `prompt_perplexity_signal`; local prompt-token logprobs via adapter forward pass; API has no prompt logprobs ⇒ neutral 0.0 (documented honest fallback) |
| Quoted-title (s4) | **IMPLEMENTED** | `quoted_title_risk` |
| s1..s4 sigmoid aggregate | **IMPLEMENTED** | `compute_prompt_risk` (weights in `config/weights.json → prompt_risk`) |

### Stage 2 — token/internal signals
| Signal | Status | Function |
|---|---|---|
| Entropy, mean logprob, logprob variance, confidence decay | **IMPLEMENTED** (pre-existing) | [logprob_metrics.py](backend/app/metrics/logprob_metrics.py), [d.py](backend/app/metrics/d.py), [hri_engine.py](backend/app/risk/hri_engine.py)`._token_entropy` |
| Semantic energy | **IMPLEMENTED + corrected** | `internal_metrics.semantic_energy` |
| Representation Collapse (RCD) | **IMPLEMENTED** (pre-existing) | `internal_metrics.representation_collapse` |
| **Logit Lens Divergence** | **IMPLEMENTED** | `internal_metrics.logit_lens_divergence_token` + `mean_logit_lens_divergence`; adapter now returns the full per-layer stack + unembed |
| **Confidence-Stability Gap** | **IMPLEMENTED** | `internal_metrics.confidence_stability_gap` + `mean_confidence_stability_gap` |

### Stage 3 — post-generation semantic checks ([semantic_checks.py](backend/app/metrics/semantic_checks.py))
| Signal | Status | Function |
|---|---|---|
| Semantic novelty | **IMPLEMENTED** | `semantic_novelty` (SBERT; injectable embedder) |
| Entity expansion | **IMPLEMENTED** | `entity_expansion` |
| Claim density | **IMPLEMENTED** (heuristic default) / **PARTIAL** (true SVO) | `claim_density` — dependency-free heuristic default; real SVO via `parser="spacy"` (needs `en_core_web_sm`) |

### HRI / learning / evaluation
| Item | Status | Where |
|---|---|---|
| HRI local & API formulas, sigmoid | **IMPLEMENTED, config-driven** | [hri_engine.py](backend/app/risk/hri_engine.py) |
| **Fitted (logistic-regression) weights** | **IMPLEMENTED** | [fit_weights.py](eval/fit_weights.py) writes `config/weights.json`; `hri_from_features` consumes it |
| Train/test split ≥1000 each, disjoint | **IMPLEMENTED** | [build_subset.py](eval/build_subset.py)`--split` |
| **TruthfulQA** loader + grading path | **IMPLEMENTED, NOT RUN** | `build_subset.py --dataset truthfulqa`; graded by `match.py` |
| Baselines: Semantic Entropy, P(True), Verbalized, text-artifact | **IMPLEMENTED** | [eval/baselines/](eval/baselines/), [run_baselines.py](eval/run_baselines.py) |
| Bootstrap CI, DeLong, ECE/Brier/reliability | **IMPLEMENTED** | [stats.py](eval/stats.py), wired into [analyze.py](eval/analyze.py) |
| Ablation (LOO + single-signal) | **IMPLEMENTED** | [ablation.py](eval/ablation.py) |
| Cross-model transfer | **IMPLEMENTED** | [score_features.py](eval/score_features.py) (+ `fit_weights` on model A) |
| Multi-seed runs | **IMPLEMENTED** | `run_benchmark --seeds` |
| Intervention false-suppression rate | **PARTIAL** | `intervened` column is logged; the metric (P(intervened | match)) is a one-line analyze addition — see §4 |

---

## 2. Still MISSING / PARTIAL — be explicit
- **Stage 1 s3 prompt perplexity in API mode**: not computable (OpenAI chat API
  returns no prompt-token logprobs). Wired as neutral 0.0 with a documented
  fallback, not faked.
- **Claim density true SVO**: spaCy is not installed in this environment, so the
  default is a documented heuristic (sentences + quantitative mentions / tokens).
  `parser="spacy"` implements real dependency-parsed SVO once `en_core_web_sm`
  is installed.
- **Entity rarity / expansion graded frequency**: default uses a bundled
  common-words list (binary proxy). Install `wordfreq` and set
  `frequency_source="wordfreq"` for graded zipf rarity. Direction holds either way.
- **Stage 1 / Stage 3 signals are not yet folded into the scalar HRI**: they are
  computed, surfaced, and logged as feature columns so `fit_weights` can learn a
  combination over the full set. The live `compute_hri` still scores from the
  Stage-2 feature block; extending the fitted feature set to all stages is a
  config/feature-list change (no new signals needed).
- **Intervention-evaluation metric**: data is logged; the summary statistic is
  not yet a function (see §4 backlog).

## 3. Exact commands the HUMAN must run (no numbers produced by code review)

Activate the env first (`conda activate ml`, or the project venv). API mode
needs `OPENAI_API_KEY` in `.env`.

```bash
# 0) sanity
python3 -m pytest tests/          # expect: all pass

# 1) Build DISJOINT train/test splits (>=1000 each)
python3 eval/build_subset.py --split --n-train 1000 --n-test 1000
python3 eval/build_subset.py --dataset truthfulqa --split --n-train 400 --n-test 400

# 2) Run the benchmark DETERMINISTICALLY on each split (local example)
python3 eval/run_benchmark.py --mode local --eval-deterministic \
    --dataset-file eval/datasets/halueval_qa_train.json \
    --out eval/results_local_train.csv
python3 eval/run_benchmark.py --mode local --eval-deterministic \
    --dataset-file eval/datasets/halueval_qa_test.json \
    --out eval/results_local_test.csv
#   (repeat with --mode api for the API model)

# 3) Grade answers (SBERT). Justify the threshold first:
python3 eval/grader_sensitivity.py --csv eval/results_local_test.csv
python3 eval/match.py --csv eval/results_local_train.csv     # uses config/grader.json
python3 eval/match.py --csv eval/results_local_test.csv

# 4) FIT HRI weights on TRAIN only, apply to config:
python3 eval/fit_weights.py --csv eval/results_local_train.csv --mode local --apply
#   compare against heuristic: python3 eval/fit_weights.py --heuristic --apply

# 5) Re-score the TEST features under the fitted weights (no regeneration):
python3 eval/score_features.py --csv eval/results_local_test.csv \
    --weights config/weights.json --mode local \
    --out eval/results_local_test_fitted.csv --report

# 6) Evaluate on the HELD-OUT TEST split (AUROC+CI, ECE, Brier, ROC+reliability):
python3 eval/analyze.py --csv eval/results_local_test_fitted.csv --mode local \
    --out eval/summary_local_test.json

# 7) Baselines + DeLong comparison:
python3 eval/run_baselines.py --csv eval/results_local_test_fitted.csv --mode local \
    --which text_similarity p_true verbalized semantic_entropy \
    --out eval/results_local_test_bl.csv
python3 eval/match.py --csv eval/results_local_test_bl.csv
python3 eval/analyze.py --csv eval/results_local_test_bl.csv --mode local \
    --out eval/summary_local_test_bl.json     # prints DeLong HRI-vs-baseline

# 8) Ablation on TEST:
python3 eval/ablation.py --csv eval/results_local_test_fitted.csv --mode local \
    --out eval/ablation_local.csv

# 9) Cross-model transfer (fit on A, score B):
python3 eval/fit_weights.py --csv eval/results_A_train.csv --mode local \
    --out eval/weights_A.json                  # dry-run write
python3 eval/score_features.py --csv eval/results_B_test.csv \
    --weights eval/weights_A.json --mode local \
    --out eval/results_B_transferA.csv --report

# 10) Second model end-to-end (Qwen2.5-3B): set the env and re-run 2–8
LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct python3 eval/run_benchmark.py --mode local \
    --eval-deterministic --dataset-file eval/datasets/halueval_qa_test.json \
    --out eval/results_qwen_test.csv

# Multi-seed variance (stochastic decoding, omit --eval-deterministic):
python3 eval/run_benchmark.py --mode local --seeds 1 2 3 \
    --dataset-file eval/datasets/halueval_qa_test.json --out eval/results_local_seeds.csv
```

Optional graded backends (recommended before reporting Stage 1/3 numbers):
```bash
pip install spacy wordfreq && python3 -m spacy download en_core_web_sm
```

## 4. What I could NOT build, and why
- **No experiments were run** and no result/summary files were written — that is
  the human's job per the project rules; this branch is capability only.
- **API-mode prompt perplexity (s3)**: impossible without prompt-token logprobs
  from the OpenAI chat API. Left as a documented neutral fallback.
- **True SVO claim density & graded entity rarity**: require `spaCy` /
  `wordfreq`, absent in this environment. Implemented as documented dependency-
  free approximations with optional upgraded backends behind a flag.
- **Folding Stage 1/3 into the live scalar HRI**: deliberately deferred to
  `fit_weights` over the logged feature columns rather than hand-weighting, so
  the combination is learned, not guessed.

## 5. One-line follow-ups (small, mechanical)
- Add `false_suppression_rate = P(intervened | model_match=="match")` to
  `eval/analyze.py` (the `intervened` column already exists).
- Extend the fitted feature set in `config/weights.json` + `compute_hri`'s
  feature dict to include Stage 1/3 + LLD/CSG once you decide which to learn over.
