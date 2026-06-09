"""Configuration, device/dtype resolution, and seed plumbing for HalluScope-v2.

Single source of truth for *where the model runs* and *in what precision*.
Per CLAUDE.md: fp16 on CUDA, 4-bit NF4 (3B) only on CUDA via bitsandbytes,
fp32 fallback on CPU for smoke runs. The 4-bit path RAISES on CPU rather than
silently degrading — we refuse to pretend a 4-bit run happened on CPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# Models sanctioned by CLAUDE.md. 0.5B is smoke-ONLY.
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_15B = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"

SEEDS = (42, 43, 44)
TOPK_LOGPROBS = 100  # CLAUDE.md: store top-100 logprobs per token, never full vocab

REPO_ROOT = Path(__file__).resolve().parents[2]


def cuda_available() -> bool:
    """True iff torch reports a usable CUDA device. Cheap, import-guarded."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclasses.dataclass
class HV2Config:
    """Resolved run configuration. ``device``/``dtype`` are derived, not guessed."""

    model_name: str = SMOKE_MODEL
    quantization: str = "none"  # "none" | "fp16" | "nf4"
    max_new_tokens: int = 48  # CLAUDE.md hard ceiling
    batch_size: int = 4  # CLAUDE.md hard ceiling
    topk_logprobs: int = TOPK_LOGPROBS
    seed: int = 42
    temperature: float = 0.7  # for the temperature probes
    # paths (relative to repo root)
    cache_dir: str = "data/cache"
    data_dir: str = "data"
    results_dir: str = "results"
    logs_dir: str = "logs"

    # ---- derived ----
    @property
    def device(self) -> str:
        return "cuda" if cuda_available() else "cpu"

    @property
    def torch_dtype(self):
        import torch

        if self.device == "cuda":
            return torch.float16
        return torch.float32  # CPU smoke: fp16 matmul is unsupported on most CPUs

    def resolve_quantization(self) -> str:
        """Return the *effective* quantization, refusing impossible combos.

        Raises if NF4 is requested without CUDA (CLAUDE.md: 4-bit needs CUDA;
        we never fake it on CPU).
        """
        if self.quantization == "nf4" and self.device != "cuda":
            raise RuntimeError(
                "4-bit NF4 quantization requires CUDA (bitsandbytes). "
                "No CUDA device is available — refusing to run a fake CPU 4-bit "
                "pass. Load the NVIDIA driver (see BLOCKERS.md B1) or use the "
                "fp16/fp32 path with a smaller model."
            )
        return self.quantization

    def abs_path(self, attr: str) -> Path:
        p = Path(getattr(self, attr))
        return p if p.is_absolute() else REPO_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["device"] = self.device
        d["cuda_available"] = cuda_available()
        return d

    def snapshot_hash(self) -> str:
        """Stable hash of the config for results provenance."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | os.PathLike[str] | None = None, **overrides: Any) -> HV2Config:
    """Load YAML config (optional) then apply keyword overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    fields = {f.name for f in dataclasses.fields(HV2Config)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return HV2Config(**data)


def set_all_seeds(seed: int) -> None:
    """Fix python/numpy/torch RNGs. Call once per process before any sampling."""
    if seed not in SEEDS:
        # Not fatal, but loud: CLAUDE.md fixes seeds to 42/43/44.
        print(f"[hv2] WARNING: seed {seed} not in sanctioned set {SEEDS}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if cuda_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
