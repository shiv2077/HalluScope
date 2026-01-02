"""
Metrics Report Module.

This module provides utilities for generating evaluation reports
and visualizing metrics from hallucination detection benchmarks.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class MetricsReport:
    """Container for evaluation metrics report."""
    title: str
    timestamp: str
    model: str
    dataset: str
    metrics: Dict[str, float]
    breakdown: Dict[str, Dict[str, float]]
    summary: str


def compute_binary_metrics(
    predictions: List[int],
    labels: List[int]
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        predictions: Predicted labels (0/1).
        labels: Ground truth labels (0/1).
        
    Returns:
        Dictionary with accuracy, precision, recall, f1.
    """
    # TODO: Implement binary metrics
    if not predictions or not labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_ranking_metrics(
    scores: List[float],
    labels: List[int]
) -> Dict[str, float]:
    """
    Compute ranking metrics (AUROC, AUPRC).
    
    Args:
        scores: Predicted risk scores.
        labels: Ground truth labels.
        
    Returns:
        Dictionary with AUROC, AUPRC.
    """
    # TODO: Implement ranking metrics
    # Use sklearn.metrics.roc_auc_score, average_precision_score
    return {
        "auroc": 0.0,
        "auprc": 0.0,
    }


def generate_report(
    predictions: List[Dict],
    model_name: str,
    dataset_name: str,
    output_path: Optional[str] = None
) -> MetricsReport:
    """
    Generate comprehensive metrics report.
    
    Args:
        predictions: List of prediction dictionaries.
        model_name: Model identifier.
        dataset_name: Dataset identifier.
        output_path: Optional path to save report.
        
    Returns:
        MetricsReport object.
    """
    # TODO: Implement report generation
    from datetime import datetime
    
    report = MetricsReport(
        title=f"HalluScope Evaluation Report",
        timestamp=datetime.now().isoformat(),
        model=model_name,
        dataset=dataset_name,
        metrics={},
        breakdown={},
        summary="",
    )
    
    # Compute metrics
    # Generate summary
    
    if output_path:
        save_report(report, output_path)
    
    return report


def save_report(report: MetricsReport, path: str) -> None:
    """
    Save report to file.
    
    Args:
        report: MetricsReport object.
        path: Output file path.
    """
    # TODO: Implement report saving
    with open(path, 'w') as f:
        json.dump({
            "title": report.title,
            "timestamp": report.timestamp,
            "model": report.model,
            "dataset": report.dataset,
            "metrics": report.metrics,
            "breakdown": report.breakdown,
            "summary": report.summary,
        }, f, indent=2)


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Format metrics as a table string.
    
    Args:
        metrics: Dictionary of metrics.
        
    Returns:
        Formatted table string.
    """
    # TODO: Implement table formatting
    lines = ["| Metric | Value |", "|--------|-------|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value:.4f} |")
    return "\n".join(lines)
