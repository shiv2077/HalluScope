"""Thermal pacing for long GPU passes on a laptop: pause when the GPU is hot, resume when it has cooled.

Sustained work at the thermal limit drops the RTX 3060 laptop clock from 2,100 MHz to about 210 MHz, so
pausing to cool loses little throughput and keeps the hardware out of its slowdown state.
"""

from __future__ import annotations

import subprocess
import sys
import time

HOT_C = 85
COOL_C = 72
CHECK_EVERY = 25


def temperature() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def pace(step: int, label: str) -> None:
    """Call every row; checks the temperature every CHECK_EVERY rows and sleeps while the GPU is hot."""
    if step % CHECK_EVERY:
        return
    t = temperature()
    if t is None or t < HOT_C:
        return
    t0 = time.time()
    while t is not None and t > COOL_C:
        time.sleep(15)
        t = temperature()
    print(f"{label}: paused {time.time() - t0:.0f}s to cool to {t}C", file=sys.stderr, flush=True)
