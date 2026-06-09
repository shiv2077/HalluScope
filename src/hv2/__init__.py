"""HalluScope-v2: observability hierarchy for hallucination detection.

Build/run order: config -> gpulock -> hdf5_store -> harness -> grading ->
datasets_hv2 -> splits -> probes -> perturb -> strata -> baselines.

HARD RULES live in repo-root CLAUDE.md. This package never stores full-vocab
logits, never fabricates results, and gates all CUDA work behind ``gpu.lock``.
"""

__all__ = ["config", "gpulock", "hdf5_store", "harness", "grading"]
