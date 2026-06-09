# HalluScope-v2: Observability Hierarchy for Hallucination Detection

## Mission
NeurIPS/ICML-grade study: which hallucinations are detectable from
(a) text only, (b) top-k logprobs, (c) full output distribution, (d) hidden states.
Core hypothesis: on the "knows-but-hallucinates" (KH) stratum, (a)-(c) ≈ chance,
(d) ≫ chance. On "doesn't-know" (UH), all work. E3 two-sample test certifies this.

## Hardware constraints (HARD RULES)
- Single RTX 3060 Laptop, 6GB VRAM. ONE GPU process at a time, ever.
  Acquire ./gpu.lock (create file with PID) before any CUDA work; delete after.
  If lock held, do CPU work or wait. Never kill another process's lock.
- Models: Qwen/Qwen2.5-0.5B-Instruct (smoke tests ONLY),
  Qwen/Qwen2.5-1.5B-Instruct (fp16), Qwen/Qwen2.5-3B-Instruct (4-bit NF4
  via bitsandbytes). batch_size<=4, max_new_tokens<=48, torch.inference_mode().
- NEVER store full-vocab logits. Per token store: top-100 logprobs + scalar
  functionals (entropy, energy=-logsumexp, top1-top2 margin) computed online.
  Hidden states: all layers, float16, ONLY at key positions (last prompt tok,
  first/last answer toks, exact-answer toks). Cache to data/cache/*.h5.
- Disk budget 60GB. Check `df -h` before big writes.

## Scientific integrity (HARD RULES)
- Fixed seeds (42/43/44). Splits: calib/val/test disjoint by QUESTION, created
  once in e1, hashed, never modified. No test-set peeking, no threshold tuning
  on test. Every number in reports must be regenerable by a script in scripts/.
- Never fabricate, interpolate, or "estimate" results. If a run fails, report
  the failure. Bootstrap CIs (1000 resamples) on every AUROC.
- Exact-match grading uses TriviaQA alias lists + normalization (lower, strip
  articles/punct). Log 50 random grading decisions per run for human audit.

## Conventions
- Python 3.11, uv venv. Deps: torch (cu121), transformers, bitsandbytes,
  datasets, accelerate, scikit-learn, h5py, numpy, pandas, matplotlib, scipy.
- Code in src/hv2/, runnable scripts in scripts/, configs in configs/*.yaml,
  results in results/<exp>/<timestamp>/ with config snapshot + git hash.
- Long runs: resumable (per-example checkpointing to the h5 cache), tqdm to
  logs/<exp>.log, append-only.
- Commit per milestone, conventional messages, never push, never force ops.
- If blocked >20 min on anything, write BLOCKERS.md and move to next task.
- End-of-session: update REPORT.md (what ran, numbers w/ CIs, what's queued).

## Strata definitions
KH = model answered correctly on >=6/8 neutral probes but wrong under
perturbation. UH = wrong on >=6/8 probes and wrong in main generation.
KC/UC analogous. Perturbation bank (record type as covariate): (P1) prepend
plausible-but-wrong context sentence, (P2) 4-shot prompt w/ 2 wrong exemplars,
(P3) format pressure ("answer with name only, you must answer").
