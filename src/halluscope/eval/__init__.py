"""
Evaluation module for HalluScope.

This module provides evaluation pipelines, metrics reporting,
and label utilities for benchmarking hallucination detection.
"""

from .run_eval import run_evaluation, EvaluationPipeline
from .metrics_report import generate_report, MetricsReport
from .label_utils import load_labels, save_predictions

__all__ = [
    "run_evaluation",
    "EvaluationPipeline",
    "generate_report",
    "MetricsReport",
    "load_labels",
    "save_predictions",
]
