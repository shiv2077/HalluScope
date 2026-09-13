# BLOCKERS — HalluScope-v2 (E0/E1 overnight run)

Last updated: 2026-06-09 (autonomous session)

## B1 — [CRITICAL, RECOVERABLE] NVIDIA driver not loaded → no CUDA

**Impact:** Every GPU stage is blocked. `torch.cuda.is_available() == False`.
This blocks: the Stage-0 VRAM ceiling test, the Stage-1 knowledge-labeling
GPU pass (1.5B fp16, 3B 4-bit), and the Stage-2 perturbed-generation capture
at scale. Per CLAUDE.md ("if a run fails, report the failure; never fabricate")
**no GPU numbers were produced or estimated.**

**Diagnosis (evidence collected this session):**
| Check | Result |
|---|---|
| `lspci` GPU present | YES — `01:00.0 NVIDIA GA106M [GeForce RTX 3060 Mobile]` |
| `nvidia-smi` | FAILS — "couldn't communicate with the NVIDIA driver" |
| `/proc/driver/nvidia` | absent |
| `lsmod \| grep nvidia` | **nvidia NOT loaded** |
| `modinfo nvidia` | module IS built: `/lib/modules/6.17.0-29-generic/updates/dkms/nvidia.ko.zst` |
| kernel | `6.17.0-29-generic` |
| `torch.cuda.is_available()` | `False` (torch 2.11.0+cu130 installed, CPU path works) |

**Root cause (CONFIRMED 2026-06-10): Secure Boot MOK key mismatch.**
`modprobe nvidia` fails with `Key was rejected by service` (`-EKEYREJECTED`).
Secure Boot is **enabled**. The DKMS module is signed, but with a key that is
**not enrolled** in firmware:
- `nvidia.ko` signer (`modinfo`): `CN=kratos Secure Boot Module Signature key`
- on disk to enroll: `/var/lib/shim-signed/mok/MOK.der` → `CN=kratos Secure Boot
  Module Signature key` (created Jun 9 20:41) — **the matching key**
- currently enrolled MOK: `CN=shiv2077-ROG-Zephyrus-G14-GA401 …` (the OLD key)

A new DKMS/MOK signing key was generated (named after the current hostname
`kratos`) and used to sign the module, but only the previous host's key is
enrolled. The kernel therefore refuses the unsigned-to-it module.

**Fix (requires user: sudo + reboot + one interactive firmware screen):**
```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der   # set a one-time password
sudo reboot
# At boot a blue "MOK Management" screen appears:
#   Enroll MOK -> Continue -> Yes -> (enter the one-time password) -> Reboot
nvidia-smi            # after reboot: lists the RTX 3060
lsmod | grep nvidia   # module now loaded
```
Alternative (less secure, no reboot dance): disable Secure Boot in BIOS.
After `nvidia-smi` works, **no code changes needed** — the harness auto-detects
CUDA and the GPU stages become a one-command launch (REPORT.md §Resume).

## B2: [RESOLVED] Missing Python deps
datasets, h5py, pandas, scikit-learn, scipy, matplotlib, accelerate, pyarrow were
absent from `.venv`. Installed this session via `uv pip install`. `bitsandbytes`
intentionally **not** installed: 4-bit NF4 needs CUDA, and importing bnb without a
GPU is fragile — the harness imports it lazily and only on the GPU/4-bit path.

## B3 — [INFO] Repo already contains HalluScope-v1
`backend/`, `frontend/`, `eval/` are an existing, unrelated hallucination-risk
project. HalluScope-v2 is built in **new** dirs (`src/hv2/`, `scripts/`,
`configs/`, `data/cache/`, `results/`) on branch `hv2/e0-e1` so v1 is untouched.

## B2: [CRITICAL, EXTERNAL] OPENAI_API_KEY rejected (routing build, 2026-09-14)

**Impact:** Phases 1 to 3 of the routing build (generation, judging, scores, gates,
calibration, evaluation) cannot run. No data was simulated. Phase 0 and Phase 4 code and docs are
complete and committed; the pipeline is verified on synthetic files only.

**Evidence:** `openai.OpenAI().models.list()` returns `401 invalid_api_key` for the key in `.env`
(prefix `sk-proj-`, suffix `lJcA`). Checked at the start and again mid-session. No
`ANTHROPIC_API_KEY` exists and none is needed: both judges are OpenAI models.

**Fix:** put a valid key in `.env`, then `nohup scripts/run_routing_phase1.sh > /dev/null 2>&1 &`.
The runner retries every 10 minutes for 8 hours and re-reads `.env` on each attempt. See
`HANDOFF.md`.
