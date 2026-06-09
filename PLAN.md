# HalluScope-v2 — E0 + E1 Implementation Plan

> **For agentic workers:** execute task-by-task; steps use checkbox (`- [ ]`) syntax.
> Obey `CLAUDE.md` HARD RULES (GPU lock, no full-vocab logits, fixed seeds,
> frozen question-disjoint splits, no fabricated numbers, bootstrap CIs).

**Goal:** Build and validate the observability-capture harness (E0) and the
knowledge-labeling + perturbed-generation data pipeline (E1/E2) for the
"observability hierarchy" study, so the heavy GPU job is a one-command launch.

**Architecture:** A single inference harness loads one Qwen2.5 model under
`gpu.lock`, and for each prompt emits a `GenRecord` capturing four nested
observation levels — text, top-100 logprobs, online scalar functionals
(entropy / energy / top1-top2 margin), and hidden states at key positions —
streamed to a resumable HDF5 cache. On top of it: knowledge probing (8 probes/Q
→ know-score), perturbed main generations (P1/P2/P3), alias-EM grading, strata
assignment (KH/UH/KC/UC), and per-access-class LR baselines.

**Tech Stack:** Python 3.11, torch 2.11 (CPU tonight; CUDA when driver back),
transformers 5.7, datasets, h5py, scikit-learn, scipy, numpy, pandas, matplotlib.

---

## EXECUTION REALITY (2026-06-09)

The NVIDIA driver is not loaded (see `BLOCKERS.md` B1) → **no CUDA tonight.**
Consequences for this plan:

| Stage | GPU-bound? | Tonight |
|---|---|---|
| S0 harness code + HDF5 + grading | No | **BUILD + CPU-smoke green** |
| S0 VRAM 5.5GB ceiling assert | Yes (needs CUDA) | replaced by **CPU peak-RAM** assert, CUDA assert skipped + logged |
| S1 dataset sampling + frozen splits | No | **BUILD + RUN (real artifacts)** |
| S1 knowledge labeling full pass (6000 Q × 8 probes, 1.5B/3B) | Yes (scale) | **code ready, GPU-gated**; CPU tiny-N validation only |
| S2 perturbed capture full | Yes (scale) | **code ready, GPU-gated**; CPU tiny-N validation only |
| S3 LR baselines | No (but needs cached features) | **code + plumbing test on smoke features** |

Smoke model = `Qwen/Qwen2.5-0.5B-Instruct` (CLAUDE.md: 0.5B = smoke ONLY).
Nothing from a smoke run is reported as a scientific result.

---

## File Structure

```
src/hv2/
  __init__.py
  config.py        # dataclass config, YAML load, seed plumbing, device/dtype resolve
  gpulock.py       # acquire/release ./gpu.lock (PID file) context manager
  hdf5_store.py    # H5Store: write/read GenRecord, per-example resume (key existence)
  harness.py       # load_model(), generate() -> GenRecord (the 4 observation levels)
  grading.py       # normalize(), alias_em(), audit-log 50 random decisions
  datasets_hv2.py  # sample TriviaQA + NQ-open (fixed seed) -> questions table
  splits.py        # question-disjoint calib/val/test, hashed + frozen
  probes.py        # 8 knowledge probes/Q (4 paraphrase greedy + 4 temp-0.7)
  perturb.py       # P1/P2/P3 perturbation bank
  strata.py        # know-score -> KH/UH/KC/UC assignment
  baselines.py     # per-access-class feature matrices + LR + bootstrap AUROC CI
configs/
  e1.yaml          # model, N, seeds, paths, thresholds
scripts/
  run_e1_knowledge.py   # GPU-gated long job: knowledge labels
  run_e2_perturbed.py   # GPU-gated long job: perturbed captures + strata
  run_s3_baselines.py   # CPU: LR preview from cached features
  smoke_e0.py           # callable CPU smoke (also invoked by the pytest)
tests/hv2/
  test_hdf5_store.py     # round-trip + resume (no model, fast)
  test_grading.py        # alias EM + normalization (no model, fast)
  test_splits.py         # disjointness + determinism (no model, fast)
  test_harness_smoke.py  # 0.5B CPU smoke: shapes, functionals, hidden states, resume
data/cache/*.h5          # observability cache (gitignored)
data/*.parquet           # questions, knowledge labels, strata (gitignored if large)
results/<exp>/<ts>/      # config snapshot + git hash + plots/tables
logs/<exp>.log           # append-only tqdm/structlog
```

---

## Task 0 — Scaffold + config + gpu.lock  *(no model)*
**Files:** create `src/hv2/__init__.py`, `src/hv2/config.py`, `src/hv2/gpulock.py`, `configs/e1.yaml`, `.gitignore` additions.
- [ ] config.py resolves device: `cuda` iff `torch.cuda.is_available()` else `cpu`; dtype fp16 on cuda, fp32 on cpu; refuses 4-bit unless cuda (raises documented error).
- [ ] gpulock.py: context manager writes PID to `./gpu.lock`, refuses if a *live* PID holds it, deletes on exit; **no-op on cpu device** (documented: lock only guards CUDA).
- [ ] commit `feat(hv2): scaffold config + gpu lock`.

## Task 1 — HDF5 store with resume  *(TDD, no model)*
**Files:** `src/hv2/hdf5_store.py`, `tests/hv2/test_hdf5_store.py`.
- [ ] test: write a synthetic GenRecord (text, top100 logprobs [T,100], token ids, functionals [T,3], hidden [P,L,H] fp16, answer-tok idx); read back; assert dtype/shape/values; assert `has(key)` true; re-open and assert resume skips existing key.
- [ ] implement H5Store (one group per example key=`{model}/{qid}/{kind}`), float16 for hidden, gzip for logprobs.
- [ ] tests green → commit `feat(hv2): resumable hdf5 store`.

## Task 2 — Grading (alias EM)  *(TDD, no model)*
**Files:** `src/hv2/grading.py`, `tests/hv2/test_grading.py`.
- [ ] test: normalize lowercases/strips punct/articles; `alias_em("The Eiffel Tower", ["eiffel tower","la tour eiffel"])` true; substring-contains rule for short gold; 50-decision audit log written.
- [ ] implement; green → commit `feat(hv2): alias-EM grader + audit log`.

## Task 3 — Harness generate() → GenRecord  *(then CPU smoke)*
**Files:** `src/hv2/harness.py`, `scripts/smoke_e0.py`, `tests/hv2/test_harness_smoke.py`.
- [ ] `load_model(name)`: one load, eval, device/dtype from config; returns (model,tok).
- [ ] `generate(model,tok,prompt,...)`: `torch.inference_mode()`, max_new_tokens<=48; per step compute top-100 logprobs + entropy(full-softmax, online) + energy(−logsumexp) + top1-top2 margin; capture hidden_states at key positions (last prompt tok, first/last answer tok, exact-answer toks) all layers fp16; locate exact-answer token indices by substring match on decoded answer.
- [ ] smoke (0.5B, 5–8 TriviaQA Qs): assert text non-empty, logprobs shape [T,100], functionals finite, hidden [P,L,H], HDF5 round-trip + resume; record peak RAM; **CUDA ceiling assert skipped with logged reason when cpu**.
- [ ] green → commit `feat(hv2): observability harness + e0 smoke green`.

## Task 4 — E1 dataset sampling + frozen splits  *(real CPU artifacts)*
**Files:** `src/hv2/datasets_hv2.py`, `src/hv2/splits.py`, `tests/hv2/test_splits.py`, `scripts/build_e1_questions.py`.
- [ ] sample 4000 TriviaQA (rc.nocontext, validation) + 2000 NQ-open (validation), seed 42; normalize to `{qid, question, gold_aliases, source}`.
- [ ] question-disjoint calib/val/test (0.4/0.3/0.3) by hashing qid; write `data/e1_questions.parquet` + `data/e1_splits.json` with a content hash; test asserts disjoint + deterministic across re-run.
- [ ] green → commit `feat(hv2): e1 question sample + frozen splits`.

## Task 5 — Probes + perturbations + strata  *(code; GPU-gated run)*
**Files:** `src/hv2/probes.py`, `src/hv2/perturb.py`, `src/hv2/strata.py`, `scripts/run_e1_knowledge.py`, `scripts/run_e2_perturbed.py`.
- [ ] probes.py: 4 paraphrase templates (greedy) + 4 temp-0.7 samples of base template = 8 probes; know-score = #correct/8.
- [ ] perturb.py: P1 prepend wrong-context sentence, P2 4-shot w/ 2 wrong exemplars, P3 format pressure; each tags `perturb_type` covariate.
- [ ] strata.py: KH = know-score≥6/8 AND perturbed-wrong; UH = know-score≤2/8 AND main-wrong; KC/UC analogous; counts per perturb type.
- [ ] run scripts: acquire gpu.lock, resumable over splits, tqdm→logs, write parquet; **guard: if not cuda → run only `--smoke-n` tiny CPU subset and label output `smoke`**.
- [ ] CPU tiny-N validation green → commit `feat(hv2): probes+perturb+strata, GPU-gated runners`.

## Task 6 — S3 baseline preview  *(CPU; plumbing-tested)*
**Files:** `src/hv2/baselines.py`, `scripts/run_s3_baselines.py`.
- [ ] feature matrices per access class: text (len, refusal flags), topk (mean/var of top-k logprob gaps), dist (entropy/energy/margin aggregates), hidden (mean-pooled last-layer at answer toks).
- [ ] LR per (access_class × stratum) on calib→val only; AUROC + 1000-resample bootstrap CI; save `results/s3_preview/*`.
- [ ] plumbing test on smoke features (synthetic if needed) → commit `feat(hv2): s3 LR baseline preview`.

---

## Decisions log (conservative-by-default, per CLAUDE.md §"When unsure")
- **DECISION(2026-06-09):** chose to build + CPU-smoke instead of faking GPU runs; no result numbers produced. Why: HARD RULE "report failures, never fabricate."
- **DECISION(2026-06-09):** new dirs `src/hv2/` etc. on branch `hv2/e0-e1`; v1 (`backend/`,`eval/`) untouched. Why: isolation, reversibility.
- **DECISION(2026-06-09):** Stage-0 VRAM-ceiling assert downgraded to CPU peak-RAM + skipped-CUDA-assert-with-log when no GPU. Why: can't measure VRAM without CUDA; refuse to invent a number.
- **DECISION(2026-06-09):** `bitsandbytes` not installed; imported lazily only on cuda/4-bit path. Why: bnb import is unreliable without CUDA.
- **DECISION(2026-06-09):** gpu.lock is a no-op on cpu device (lock guards CUDA contention only). Why: lock's purpose is single-GPU-process safety; CPU has no such constraint.

## Self-review (spec coverage)
- text/top-k/dist/hidden capture → Task 3 ✓  · resumable HDF5 → Task 1,3 ✓
- 4000 TriviaQA + 2000 NQ, seed → Task 4 ✓ · frozen disjoint splits + hash → Task 4 ✓
- 8 probes (4 paraphrase+4 temp) → Task 5 ✓ · P1/P2/P3 → Task 5 ✓
- KH/UH/KC/UC + counts/perturb → Task 5 ✓ · KH<300 reporting → REPORT.md ✓
- LR per access class + bootstrap CI → Task 6 ✓ · gpu.lock/seeds/no-fabrication → throughout ✓
