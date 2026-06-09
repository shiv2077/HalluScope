"""Single-GPU mutex via ``./gpu.lock`` (CLAUDE.md HARD RULE).

Only ONE GPU process may run at a time on the 6GB RTX 3060. Acquire the lock
(a PID file at repo root) before any CUDA work; release on exit. Never kill
another process's lock — if a *live* PID holds it, we wait or refuse.

On CPU-only devices the lock is a no-op: it guards GPU-memory contention, and
there is none on CPU. This keeps smoke runs from blocking on a stale lock.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .config import REPO_ROOT, cuda_available

LOCK_PATH = REPO_ROOT / "gpu.lock"


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` currently exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def _read_lock() -> int | None:
    try:
        txt = LOCK_PATH.read_text().strip()
        return int(txt.split()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return None


class GpuLock:
    """Context manager. ``with GpuLock(): ...`` around all CUDA work.

    Args:
        wait: if True, block until the lock frees instead of raising.
        poll_s: poll interval while waiting.
        timeout_s: max seconds to wait before raising (None = forever).
    """

    def __init__(self, wait: bool = True, poll_s: float = 5.0, timeout_s: float | None = None):
        self.wait = wait
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.active = cuda_available()  # no-op on CPU
        self._held = False

    def __enter__(self) -> "GpuLock":
        if not self.active:
            print("[gpu.lock] CPU device — lock is a no-op (no GPU contention).")
            return self
        deadline = None if self.timeout_s is None else time.time() + self.timeout_s
        while True:
            holder = _read_lock()
            if holder is None or not _pid_alive(holder):
                if holder is not None:
                    print(f"[gpu.lock] clearing stale lock from dead PID {holder}")
                # Atomic-ish create; re-check to avoid a race with another waiter.
                LOCK_PATH.write_text(f"{os.getpid()} {int(time.time())}\n")
                if _read_lock() == os.getpid():
                    self._held = True
                    print(f"[gpu.lock] acquired by PID {os.getpid()}")
                    return self
            if not self.wait:
                raise RuntimeError(f"gpu.lock held by live PID {holder}; not waiting.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"gpu.lock held by PID {holder}; timed out waiting.")
            print(f"[gpu.lock] held by live PID {holder}; waiting {self.poll_s}s...")
            time.sleep(self.poll_s)

    def __exit__(self, *exc) -> None:
        if self._held and _read_lock() == os.getpid():
            try:
                LOCK_PATH.unlink()
                print(f"[gpu.lock] released by PID {os.getpid()}")
            except FileNotFoundError:
                pass
        self._held = False
