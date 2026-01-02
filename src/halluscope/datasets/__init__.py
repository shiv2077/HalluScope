"""
Datasets module for HalluScope.

This module provides dataset loading and processing utilities
for hallucination evaluation benchmarks.
"""

from .truthfulqa import load_truthfulqa, TruthfulQADataset
from .halueval import load_halueval, HaluEvalDataset
from .common import DatasetItem, preprocess_dataset

__all__ = [
    "load_truthfulqa",
    "TruthfulQADataset",
    "load_halueval",
    "HaluEvalDataset",
    "DatasetItem",
    "preprocess_dataset",
]
