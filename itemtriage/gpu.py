"""Local GPU inference helpers: a single-GPU lock file and a model loader. Imported only by the GPU passes.

The lock is a PID file at the repo root. One CUDA process runs at a time on the 6 GB card. A lock held by a
dead PID is cleared. A lock held by a live PID is waited on, never broken. On CPU the lock is a no-op.
"""

from __future__ import annotations

import os
import time

from . import paths

LOCK_PATH = paths.ROOT / "gpu.lock"
POLL_SECONDS = 5.0


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock() -> int | None:
    try:
        return int(LOCK_PATH.read_text().split()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return None


class GpuLock:
    """``with GpuLock(): ...`` around all CUDA work. wait=False raises if a live PID holds the lock."""

    def __init__(self, wait: bool = True):
        self.wait = wait
        self.active = cuda_available()
        self._held = False

    def __enter__(self) -> "GpuLock":
        if not self.active:
            return self
        while True:
            holder = _read_lock()
            if holder is None or not _pid_alive(holder):
                LOCK_PATH.write_text(f"{os.getpid()} {int(time.time())}\n")
                if _read_lock() == os.getpid():
                    self._held = True
                    return self
            if not self.wait:
                raise RuntimeError(f"gpu.lock held by live PID {holder}")
            time.sleep(POLL_SECONDS)

    def __exit__(self, *exc) -> None:
        if self._held and _read_lock() == os.getpid():
            LOCK_PATH.unlink(missing_ok=True)
        self._held = False


def load_model(name: str, quantization: str):
    """Load a causal LM and its tokenizer in eval mode. quantization: "fp16" or "nf4". NF4 needs CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = cuda_available()
    if quantization == "nf4" and not cuda:
        raise RuntimeError("4-bit NF4 needs CUDA (bitsandbytes); refusing a CPU run")
    tok = AutoTokenizer.from_pretrained(name)
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0})
    else:
        dtype = torch.float16 if cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("cuda" if cuda else "cpu")
    model.eval()
    return model, tok
