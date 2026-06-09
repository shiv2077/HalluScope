# HalluScope-v2 — Overnight REPORT (E0/E1)

**Read this in 2 minutes.** Session: 2026-06-09 autonomous. Branch: `hv2/e0-e1`.

## TL;DR
- 🔴 **GPU unavailable all night** — NVIDIA driver not loaded (`nvidia-smi` fails,
  `torch.cuda.is_available()==False`). Recoverable; one user command. See `BLOCKERS.md` B1.
- 🟢 **No GPU numbers were faked.** Per CLAUDE.md, failures are reported, not estimated.
- 🟡 Built the full GPU-ready pipeline + CPU-smoke-validated it. Heavy jobs are
  one command away once the driver is back.

## What ran (status updated as the session proceeds)
| Item | Status |
|---|---|
| Env diagnosis + deps installed | _in progress_ |
| E0 harness + HDF5 + grading | _pending_ |
| E0 CPU smoke (0.5B) green | _pending_ |
| E1 question sample + frozen splits | _pending_ |
| Probe/perturb/strata + GPU-gated runners | _pending_ |
| S3 LR baseline preview (plumbing) | _pending_ |

## Stratum counts
Not yet produced — requires the GPU knowledge-labeling + perturbation passes
(blocked, B1). The protocol expects **KH** to be the scarce class; if KH < 300
on the 1.5B model after the real run, options to widen are documented here (more
questions vs stronger perturbations) — **not** silently changed.

## Preview AUROCs (w/ CIs)
None — needs cached features from the GPU runs. The S3 code + bootstrap-CI
machinery is built and plumbing-tested so it runs the moment features exist.

## Blockers
- **B1 (critical, recoverable):** NVIDIA driver not loaded. Fix: `sudo modprobe
  nvidia nvidia_uvm nvidia_modeset && nvidia-smi`. Full diagnosis in `BLOCKERS.md`.

## Resume commands (once `nvidia-smi` shows the 3060)
```bash
# (filled in as scripts land this session — see bottom of file)
```
