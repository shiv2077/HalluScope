#!/usr/bin/env python3
"""
HalluScope External Benchmark Runner

Evaluates HalluScope's hallucination detection accuracy on:
  1) TruthfulQA (generation split)
  2) HaluEval

This script ONLY consumes outputs from the existing HalluScope inference pipeline.
It does NOT modify any existing code — only loads datasets, sends prompts,
collects HRI scores, and computes benchmark metrics.

Usage:
    conda activate ml
    
    # Run on TruthfulQA (all ~817 samples)
    python scripts/run_benchmark.py --dataset truthfulqa
    
    # Run on HaluEval (1000 random samples, seed=42)
    python scripts/run_benchmark.py --dataset halueval --limit 1000
    
    # Custom threshold
    python scripts/run_benchmark.py --dataset truthfulqa --threshold 0.6
    
    # Debug mode (first 10 samples)
    python scripts/run_benchmark.py --dataset truthfulqa --limit 10 --debug

Expected runtime on RTX 3060 with Qwen2.5-3B:
    - TruthfulQA (817): ~45 min
    - HaluEval (1000): ~55 min
    - Total (~1800): ~1.5 hours
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for HalluScope imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.70
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_INTERVAL = 10  # Save every N samples
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_truthfulqa() -> List[Dict[str, Any]]:
    """
    Load TruthfulQA generation split from HuggingFace.
    
    Returns list of dicts with:
        - prompt: question text
        - ground_truth: 1 if the question is designed to elicit a false answer, 0 otherwise
        - category: question category
        - best_answer: reference correct answer
        - incorrect_answers: list of common incorrect answers
    
    Note: TruthfulQA questions are designed to elicit false/misleading answers
    from LLMs, so most prompts should result in HIGH hallucination risk when
    the model gives a confident wrong answer.
    """
    from datasets import load_dataset
    
    print("Loading TruthfulQA (generation split)...")
    ds = load_dataset("truthful_qa", "generation", trust_remote_code=True)
    
    samples = []
    for item in ds["validation"]:
        # TruthfulQA questions are tricky — designed to elicit hallucinations
        # Ground truth: 1 = hallucination-inducing question, 0 = safe question
        # For benchmark: we expect HRI should be HIGH for these prompts
        samples.append({
            "prompt": item["question"],
            "ground_truth": 1,  # All TruthfulQA questions are hallucination traps
            "category": item.get("category", "unknown"),
            "best_answer": item.get("best_answer", ""),
            "incorrect_answers": item.get("incorrect_answers", []),
            "source": "truthfulqa",
        })
    
    print(f"  Loaded {len(samples)} samples from TruthfulQA")
    return samples


def load_halueval(limit: int = 1000, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """
    Load HaluEval dataset from HuggingFace.
    
    Returns list of dicts with:
        - prompt: input text
        - ground_truth: 1 if hallucinated, 0 if factual
        - task: QA / summarization / dialogue
    
    HaluEval contains both hallucinated and factual samples, making it
    suitable for binary classification evaluation.
    """
    from datasets import load_dataset
    
    print(f"Loading HaluEval (sampling {limit} with seed={seed})...")
    
    # HaluEval has multiple subsets — load and combine
    samples = []
    
    try:
        # Try loading different HaluEval subsets
        subsets = ["qa_samples", "summarization_samples", "dialogue_samples"]
        
        for subset in subsets:
            try:
                ds = load_dataset("pminervini/HaluEval", subset, trust_remote_code=True)
                split_name = "data" if "data" in ds else list(ds.keys())[0]
                
                for item in ds[split_name]:
                    # Determine ground truth label
                    # HaluEval marks hallucinated samples
                    is_hallucinated = 1 if item.get("hallucination", "no").lower() == "yes" else 0
                    
                    # Get prompt based on task type
                    if "question" in item:
                        prompt = item["question"]
                    elif "dialogue_history" in item:
                        prompt = item["dialogue_history"]
                    elif "document" in item:
                        prompt = f"Summarize: {item['document'][:500]}"
                    else:
                        prompt = str(item.get("input", item.get("text", "")))
                    
                    if prompt.strip():
                        samples.append({
                            "prompt": prompt,
                            "ground_truth": is_hallucinated,
                            "task": subset.replace("_samples", ""),
                            "source": "halueval",
                        })
            except Exception as e:
                print(f"  Warning: Could not load {subset}: {e}")
                continue
    
    except Exception as e:
        print(f"  Warning: Error loading HaluEval main subsets: {e}")
        
        # Fallback: try alternative loading
        try:
            ds = load_dataset("pminervini/HaluEval", trust_remote_code=True)
            for split in ds.keys():
                for item in ds[split]:
                    prompt = item.get("question", item.get("input", ""))
                    if prompt.strip():
                        samples.append({
                            "prompt": prompt,
                            "ground_truth": 1 if item.get("hallucination", "no").lower() == "yes" else 0,
                            "task": "mixed",
                            "source": "halueval",
                        })
        except Exception as e2:
            print(f"  Warning: Fallback loading also failed: {e2}")
    
    if not samples:
        raise ValueError("Could not load any samples from HaluEval. Check dataset availability.")
    
    # Random sample
    np.random.seed(seed)
    if len(samples) > limit:
        indices = np.random.choice(len(samples), size=limit, replace=False)
        samples = [samples[i] for i in indices]
    
    print(f"  Loaded {len(samples)} samples from HaluEval")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# HalluScope Interface (READ-ONLY — no modifications)
# ─────────────────────────────────────────────────────────────────────────────

class HalluScopeEvaluator:
    """
    Wrapper class for calling the existing HalluScope inference pipeline.
    
    This class ONLY consumes outputs — it does NOT modify any HalluScope code.
    """
    
    def __init__(self, model_id: str = MODEL_ID, debug: bool = False):
        self.model_id = model_id
        self.debug = debug
        self._llm = None
        
    def _load_model(self):
        """Lazy load the LLM backend."""
        if self._llm is None:
            from halluscope.core.llm_backend_v4 import LLMBackend
            print(f"\nLoading HalluScope model: {self.model_id}...")
            self._llm = LLMBackend(model_id=self.model_id, device="auto")
            print("✅ Model loaded\n")
        return self._llm
    
    def generate(self, prompt: str, max_new_tokens: int = 60) -> Dict[str, Any]:
        """
        Send prompt to HalluScope and collect results.
        
        Returns:
            dict with keys:
                - generated_text: str
                - hri: float (final HRI score)
                - risk_level: str (LOW/MEDIUM/HIGH)
                - metrics: dict (detailed signal breakdown)
                - latency_ms: float
        """
        from halluscope.app.service_v4 import analyze_prompt
        
        llm = self._load_model()
        
        start_time = time.time()
        
        result = analyze_prompt(
            llm=llm,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            hri_threshold=0.70,
            early_stop_enabled=False,  # Let it generate fully for evaluation
            debug=self.debug,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract final HRI — use the three-stage blended score
        # Stage 1 (pre-hoc) + Stage 2 (generation) + Stage 3 (post-hoc)
        stage1_risk = result.get("stage1_prehoc", {}).get("prompt_risk_score", 0.0)
        stage2_hri = result.get("stage2_generation", {}).get("generation_hri", 0.0)
        stage3_risk = result.get("stage3_posthoc", {}).get("verification_risk", 0.0)
        
        # Final HRI computation (matches service_v4.py)
        BLEND_GEN = 0.55
        BLEND_POST = 0.45
        POST_HOC_FLOOR = 0.80
        
        blended = BLEND_GEN * stage2_hri + BLEND_POST * stage3_risk
        post_floor = stage3_risk * POST_HOC_FLOOR
        pre_floor = stage1_risk * 0.5
        
        final_hri = max(blended, post_floor, pre_floor)
        final_hri = min(1.0, final_hri)
        
        # Determine risk level
        if final_hri >= 0.70:
            risk_level = "HIGH"
        elif final_hri >= 0.45:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "generated_text": result.get("answer", ""),
            "hri": final_hri,
            "risk_level": risk_level,
            "metrics": {
                "stage1_prehoc": result.get("stage1_prehoc", {}),
                "stage2_generation": result.get("stage2_generation", {}),
                "stage3_posthoc": result.get("stage3_posthoc", {}),
            },
            "latency_ms": latency_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    samples: List[Dict[str, Any]],
    evaluator: HalluScopeEvaluator,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Path = RESULTS_DIR,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    dataset_name: str = "benchmark",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run evaluation loop over all samples.
    
    Args:
        samples: List of dicts with 'prompt' and 'ground_truth' keys
        evaluator: HalluScopeEvaluator instance
        threshold: HRI threshold for binary classification
        output_dir: Directory to save results
        checkpoint_interval: Save progress every N samples
        dataset_name: Name for output files
    
    Returns:
        Tuple of (predictions_df, metrics_dict)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = []
    csv_path = output_dir / f"halluscope_{dataset_name}_predictions.csv"
    
    # Check for existing checkpoint
    start_idx = 0
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        start_idx = len(existing)
        predictions = existing.to_dict("records")
        print(f"Resuming from checkpoint: {start_idx} samples already processed")
    
    print(f"\nRunning evaluation on {len(samples)} samples (threshold={threshold})...\n")
    
    for i, sample in enumerate(tqdm(samples[start_idx:], initial=start_idx, total=len(samples))):
        try:
            # Send prompt to HalluScope
            result = evaluator.generate(sample["prompt"])
            
            # Convert HRI to binary prediction
            hri = result["hri"]
            prediction = 1 if hri >= threshold else 0
            
            predictions.append({
                "prompt": sample["prompt"][:500],  # Truncate for CSV
                "generated_text": result["generated_text"][:500],
                "hri": round(hri, 4),
                "prediction": prediction,
                "ground_truth": sample["ground_truth"],
                "latency_ms": round(result["latency_ms"], 2),
                "risk_level": result["risk_level"],
                "source": sample.get("source", "unknown"),
                "category": sample.get("category", sample.get("task", "unknown")),
            })
            
            # Incremental save
            if (start_idx + i + 1) % checkpoint_interval == 0:
                df = pd.DataFrame(predictions)
                df.to_csv(csv_path, index=False)
                tqdm.write(f"  Checkpoint saved: {len(predictions)} samples")
                
        except Exception as e:
            tqdm.write(f"  Error on sample {start_idx + i}: {e}")
            predictions.append({
                "prompt": sample["prompt"][:500],
                "generated_text": "",
                "hri": 0.0,
                "prediction": 0,
                "ground_truth": sample["ground_truth"],
                "latency_ms": 0.0,
                "risk_level": "ERROR",
                "source": sample.get("source", "unknown"),
                "category": sample.get("category", sample.get("task", "unknown")),
                "error": str(e),
            })
    
    # Final save
    df = pd.DataFrame(predictions)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Predictions saved to: {csv_path}")
    
    # Compute metrics
    metrics = compute_metrics(df, threshold, dataset_name)
    
    # Save metrics
    metrics_path = output_dir / f"halluscope_{dataset_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")
    
    return df, metrics


def compute_metrics(
    df: pd.DataFrame,
    threshold: float,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics.
    
    Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )
    
    # Filter out error rows
    valid = df[df["risk_level"] != "ERROR"].copy()
    
    if len(valid) == 0:
        return {"error": "No valid predictions"}
    
    y_true = valid["ground_truth"].values
    y_pred = valid["prediction"].values
    y_scores = valid["hri"].values
    
    # Handle case where all labels are the same
    n_classes = len(np.unique(y_true))
    
    metrics = {
        "dataset": dataset_name,
        "threshold": threshold,
        "total_samples": len(df),
        "valid_samples": len(valid),
        "error_samples": len(df) - len(valid),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Basic metrics
    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    
    # Precision, Recall, F1 (handle edge cases)
    try:
        metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
        metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
        metrics["f1_score"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
    except Exception as e:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0
        metrics["metric_error"] = str(e)
    
    # ROC-AUC (requires both classes present)
    try:
        if n_classes > 1:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_scores), 4)
        else:
            metrics["roc_auc"] = None
            metrics["roc_auc_note"] = "Cannot compute (single class in ground truth)"
    except Exception as e:
        metrics["roc_auc"] = None
        metrics["roc_auc_error"] = str(e)
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]) if cm.shape[0] > 0 else 0,
            "fp": int(cm[0, 1]) if cm.shape[1] > 1 else 0,
            "fn": int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            "tp": int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
        }
    except Exception as e:
        metrics["confusion_matrix_error"] = str(e)
    
    # Distribution stats
    metrics["hri_stats"] = {
        "mean": round(float(y_scores.mean()), 4),
        "std": round(float(y_scores.std()), 4),
        "min": round(float(y_scores.min()), 4),
        "max": round(float(y_scores.max()), 4),
        "median": round(float(np.median(y_scores)), 4),
    }
    
    # Latency stats
    latencies = valid["latency_ms"].values
    metrics["latency_stats"] = {
        "mean_ms": round(float(latencies.mean()), 2),
        "std_ms": round(float(latencies.std()), 2),
        "min_ms": round(float(latencies.min()), 2),
        "max_ms": round(float(latencies.max()), 2),
        "total_seconds": round(float(latencies.sum() / 1000), 2),
    }
    
    # Class distribution
    metrics["class_distribution"] = {
        "ground_truth_positive": int(y_true.sum()),
        "ground_truth_negative": int(len(y_true) - y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "predicted_negative": int(len(y_pred) - y_pred.sum()),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty print metrics summary."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {metrics.get('dataset', 'unknown')}")
    print("=" * 60)
    print(f"Threshold: {metrics.get('threshold', 'N/A')}")
    print(f"Samples:   {metrics.get('valid_samples', 0)} valid / {metrics.get('total_samples', 0)} total")
    print("-" * 60)
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1_score', 0):.4f}")
    print(f"ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
    print("-" * 60)
    
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        print("Confusion Matrix:")
        print(f"  TN={cm.get('tn', 0):4d}  FP={cm.get('fp', 0):4d}")
        print(f"  FN={cm.get('fn', 0):4d}  TP={cm.get('tp', 0):4d}")
    
    if "hri_stats" in metrics:
        hri = metrics["hri_stats"]
        print("-" * 60)
        print(f"HRI Stats: mean={hri['mean']:.3f} std={hri['std']:.3f} "
              f"[{hri['min']:.3f}, {hri['max']:.3f}]")
    
    if "latency_stats" in metrics:
        lat = metrics["latency_stats"]
        print(f"Latency:   mean={lat['mean_ms']:.0f}ms total={lat['total_seconds']:.1f}s")
    
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HalluScope External Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on TruthfulQA (all samples)
    python scripts/run_benchmark.py --dataset truthfulqa
    
    # Run on HaluEval (1000 random samples)
    python scripts/run_benchmark.py --dataset halueval --limit 1000
    
    # Custom threshold
    python scripts/run_benchmark.py --dataset truthfulqa --threshold 0.6
    
    # Debug mode (10 samples)
    python scripts/run_benchmark.py --dataset truthfulqa --limit 10 --debug
        """,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["truthfulqa", "halueval", "both"],
        default="truthfulqa",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (default: all for TruthfulQA, 1000 for HaluEval)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"HRI threshold for binary classification (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose output)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Output directory for results (default: {RESULTS_DIR})",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("HalluScope External Benchmark Runner")
    print("=" * 60)
    print(f"Dataset:   {args.dataset}")
    print(f"Threshold: {args.threshold}")
    print(f"Limit:     {args.limit or 'all'}")
    print(f"Debug:     {args.debug}")
    print(f"Output:    {args.output_dir}")
    print("=" * 60 + "\n")
    
    output_dir = Path(args.output_dir)
    
    # Initialize evaluator
    evaluator = HalluScopeEvaluator(debug=args.debug)
    
    all_results = []
    
    # Run on selected datasets
    if args.dataset in ["truthfulqa", "both"]:
        samples = load_truthfulqa()
        if args.limit:
            samples = samples[:args.limit]
        
        df, metrics = run_evaluation(
            samples=samples,
            evaluator=evaluator,
            threshold=args.threshold,
            output_dir=output_dir,
            dataset_name="truthfulqa",
        )
        print_metrics(metrics)
        all_results.append(("truthfulqa", df, metrics))
    
    if args.dataset in ["halueval", "both"]:
        limit = args.limit or 1000
        samples = load_halueval(limit=limit)
        
        df, metrics = run_evaluation(
            samples=samples,
            evaluator=evaluator,
            threshold=args.threshold,
            output_dir=output_dir,
            dataset_name="halueval",
        )
        print_metrics(metrics)
        all_results.append(("halueval", df, metrics))
    
    # Combined summary for "both"
    if args.dataset == "both" and len(all_results) == 2:
        print("\n" + "=" * 60)
        print("COMBINED SUMMARY")
        print("=" * 60)
        for name, df, metrics in all_results:
            print(f"\n{name.upper()}:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
            print(f"  ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
        print("=" * 60 + "\n")
    
    print("✅ Benchmark complete!")


if __name__ == "__main__":
    main()
