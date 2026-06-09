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

**Root cause:** the DKMS-built `nvidia.ko` exists for the running kernel but the
module is not loaded into the kernel. This is almost always a one-command fix;
the most common trigger is a kernel update (6.17 is recent) where the module was
rebuilt but never modprobed, or `nvidia-persistenced`/UVM not started.

**Fix (requires the user — no passwordless sudo in this session):**
```bash
sudo modprobe nvidia nvidia_uvm nvidia_modeset   # try this first
nvidia-smi                                        # should now list the 3060
# If modprobe errors, the module/userspace are out of sync after a kernel bump:
sudo apt install --reinstall nvidia-dkms-<ver> && sudo reboot
```
After `nvidia-smi` shows the GPU, **no code changes are needed** — the harness
auto-detects CUDA and the GPU stages become a one-command launch (see REPORT.md
§Resume).

## B2 — [RESOLVED] Missing Python deps
datasets, h5py, pandas, scikit-learn, scipy, matplotlib, accelerate, pyarrow were
absent from `.venv`. Installed this session via `uv pip install`. `bitsandbytes`
intentionally **not** installed: 4-bit NF4 needs CUDA, and importing bnb without a
GPU is fragile — the harness imports it lazily and only on the GPU/4-bit path.

## B3 — [INFO] Repo already contains HalluScope-v1
`backend/`, `frontend/`, `eval/` are an existing, unrelated hallucination-risk
project. HalluScope-v2 is built in **new** dirs (`src/hv2/`, `scripts/`,
`configs/`, `data/cache/`, `results/`) on branch `hv2/e0-e1` so v1 is untouched.
