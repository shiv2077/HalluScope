"""Shared seeding utility for reproducible runs.

Seeds Python ``random``, NumPy, and (if installed) PyTorch — including CUDA —
from a single call so that every experiment can be made deterministic.

Generation reproducibility note
-------------------------------
Seeding fixes sampling RNGs, but truly deterministic *decoding* in eval also
requires greedy decoding (temperature = 0 / do_sample = False). The benchmark
runner's ``--eval-deterministic`` flag combines both: it seeds via
:func:`set_global_seed` AND forces temperature = 0 in the request.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(
    seed: int = DEFAULT_SEED, *, deterministic_torch: bool = False
) -> int:
    """Seed all RNGs in use. Returns the seed for logging.

    Args:
        seed: the integer seed.
        deterministic_torch: if True, also request deterministic CUDA/cuDNN
            kernels (slower; only needed for exact GPU reproducibility).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass

    return seed
