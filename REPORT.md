# HalluScope-v2 — Overnight REPORT (E0/E1)

**2-minute read.** Session: 2026-06-09 → 06-10 autonomous. Branch: `hv2/e0-e1`
(2 commits). Everything below is regenerable by a script in `scripts/`.

## TL;DR
- 🔴 **GPU was unavailable all night** — the NVIDIA driver is not loaded
  (`torch.cuda.is_available()==False`). **Recoverable in one command.** See B1.
- 🟢 **Zero GPU numbers were fabricated.** Per CLAUDE.md, the blocked stages are
  reported as blocked, not estimated.
- 🟢 **E0 is done and green.** The observability harness (text · top-100 logprobs ·
  entropy/energy/margin · hidden states) is built and validated end-to-end on CPU.
- 🟢 **E1 data prep is done with REAL artifacts:** 6000 questions sampled + frozen
  question-disjoint splits.
- 🟡 **E1/E2/S3 heavy jobs are built, CPU-validated, and GPU-gated** — one command
  away once the driver returns.

## What ran
| Item | Status | Evidence |
|---|---|---|
| Env diagnosis + 8 deps installed | ✅ | BLOCKERS.md B1/B2 |
| E0 harness + HDF5 + grading | ✅ committed `6440ba0b` | 17 fast tests |
| E0 CPU smoke (0.5B, 5 Q) | ✅ green | `logs/smoke_e0.log`: shapes ok, RSS 3.8GB |
| **E1 question pool: 4000 TriviaQA + 2000 NQ-open = 6000** | ✅ real | `data/e1_questions.parquet` |
| **E1 frozen splits (calib 2418 / val 1818 / test 1764)** | ✅ real | hash `3f705637fcea…` |
| E1 knowledge loop | ✅ CPU-validated (5 Q) | bands {low:4, high:1} |
| E2 perturbed + strata | ✅ CPU-validated (5 Q) | strata table below |
| S3 baseline preview | ✅ plumbing green | runs end-to-end, n/a on tiny N |
| 29 fast unit tests | ✅ green | `pytest tests/hv2 -m "not slow"` |
| **E1/E2 full GPU passes (1.5B/3B)** | ⛔ blocked (B1) | refused on CPU, exit 2 |

## Stratum counts
**Real counts are NOT yet available** — they need the GPU knowledge-labeling +
perturbation passes (blocked, B1). What exists is a **0.5B CPU smoke over 5
questions** (NOT a result, only proof the strata machinery works):

```
overall: KH=1  KC=2  UH=12  UC=0  MID=0     (15 perturbed records)
by type: P1 KH=0  P2 KH=0  P3 KH=1          (P3 format-pressure produced the KH)
```

**KH scarcity (the central, expected-scarce stratum).** Protocol expectation:
KH is rare. Per your instruction, if **KH < 300** on the real 1.5B run, I will NOT
silently change the protocol. Documented widening knobs, in preference order:
1. **Stronger perturbations first** (cheap): raise P2 wrong-exemplar count 2→3,
   add a P1 variant that cites a fake authority, stack P1+P3. KH is produced by
   perturbations flipping *known* answers, so this is the highest-leverage knob.
2. **More questions** (linear cost): the pool is 6000; widen `--n-triviaqa`/`--n-nq`.
   Only ~ (high-band fraction)×(flip rate) of these become KH, so this is the
   expensive lever — use after (1).
3. Add the 3B model — a stronger model has a larger "knows" set, so more KH headroom.
Decision deferred to the real run; numbers will dictate which knob.

## Preview AUROCs (with CIs)
**None yet** — the S3 detectors need cached features from the GPU runs. The
machinery is built and unit-tested: per access class (text ⊆ topk ⊆ dist, and
hidden), LR on calib → AUROC on val with **1000-resample bootstrap CIs**,
stratified by knows/unknown subpopulation. On the smoke cache it correctly prints
`n/a` (N below the min-sample guard) rather than a fabricated number.

## Blockers
- **B1 (critical, recoverable): NVIDIA driver not loaded.** GPU present
  (`RTX 3060`), DKMS module built for kernel 6.17.0-29, just not modprobed.
  **Fix:** `sudo modprobe nvidia nvidia_uvm nvidia_modeset && nvidia-smi`.
  No passwordless sudo this session, so I could not load it. Full diagnosis: BLOCKERS.md.

## Resume — exact commands once `nvidia-smi` shows the 3060
```bash
cd /home/shiv2077/dev/HalluScope && source .venv/bin/activate   # or use .venv/bin/python

# 0. confirm CUDA is back
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"   # must be True

# 1. (already done; re-run only to rebuild) question pool + frozen splits
#    .venv/bin/python scripts/build_e1_questions.py --n-triviaqa 4000 --n-nq 2000

# 2. E1 knowledge labeling — 1.5B fp16 (the long job; resumable, ~hours)
.venv/bin/python scripts/run_e1_knowledge.py --config configs/e1.yaml 2>&1 | tee -a logs/e1.log

# 3. E2 perturbed generations + strata table
.venv/bin/python scripts/run_e2_perturbed.py --config configs/e1.yaml 2>&1 | tee -a logs/e2.log

# 4. S3 baseline preview (AUROC + bootstrap CIs, plots in results/)
.venv/bin/python scripts/run_s3_baselines.py --tag Qwen2.5-1.5B-Instruct 2>&1 | tee -a logs/s3.log

# 5. if time remains: 3B 4-bit — edit configs/e1.yaml (model_name=3B, quantization=nf4)
#    then repeat steps 2–4. NF4 auto-requires CUDA; the harness refuses it on CPU.
```
All four runners hold `gpu.lock`, checkpoint per example (resumable after any
interruption), and log to `logs/`. The frozen splits (`data/e1_splits.json`,
hash `3f705637fcea…`) are immutable — `freeze_splits` refuses to overwrite them.

## Tests
```
.venv/bin/python -m pytest tests/hv2 -m "not slow"   # 29 passed
.venv/bin/python -m pytest tests/hv2 -m slow         # 1 passed (loads 0.5B on CPU)
```

## Decisions made autonomously (conservative-by-default)
See PLAN.md §"Decisions log". Headlines: build+CPU-validate instead of faking GPU;
new `src/hv2/` tree on branch `hv2/e0-e1` (v1 untouched); VRAM-ceiling assert
downgraded to CPU-RSS + skipped-with-log; `bitsandbytes` deferred to the CUDA path;
TriviaQA sampler dedups-while-streaming so it yields the full 4000 unique questions
(naïve `islice(4000)` gave only 3614 due to repeated `question_id`s — fixed, not hidden).

## Separate track — Worktree #2 (`../hv2-paper`, paper/skeleton)
NOT started by this session (it is a parallel CPU track and, per your prompt,
depends on a related-work summary "to paste next" that did not arrive, plus
`results/` that the GPU runs will fill). It is unblocked the moment you (a) paste
that summary and (b) have any `results/*/auroc.json` to wire the table macros to —
until then the paper's result macros should render "—", exactly as its prompt asks.

## Next session should start with
1. `sudo modprobe nvidia nvidia_uvm nvidia_modeset` → verify `nvidia-smi`.
2. Run Resume steps 2–4 above (1.5B). Watch `logs/e1.log` / `logs/e2.log`.
3. Check real KH count; if <300, apply widening knob (1) and document the change.
4. Then start the 3B 4-bit pass if VRAM/time allow.
