#!/usr/bin/env python3
"""
Dataset Download Script.

This script downloads the required datasets for HalluScope
evaluation from HuggingFace.
"""

import argparse
import os
from pathlib import Path


def download_truthfulqa(output_dir: Path) -> None:
    """
    Download TruthfulQA dataset.
    
    Args:
        output_dir: Output directory for dataset.
    """
    # TODO: Implement TruthfulQA download
    print("Downloading TruthfulQA dataset...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("truthful_qa", "generation")
        
        # Save to disk
        output_path = output_dir / "truthfulqa"
        dataset.save_to_disk(str(output_path))
        
        print(f"TruthfulQA saved to {output_path}")
    except Exception as e:
        print(f"Error downloading TruthfulQA: {e}")


def download_halueval(output_dir: Path) -> None:
    """
    Download HaluEval dataset.
    
    Args:
        output_dir: Output directory for dataset.
    """
    # TODO: Implement HaluEval download
    print("Downloading HaluEval dataset...")
    
    try:
        from datasets import load_dataset
        
        # Download different tasks
        for task in ["qa", "summarization", "dialogue"]:
            print(f"  Downloading {task} subset...")
            # dataset = load_dataset("halueval", task)
            # TODO: Save dataset
        
        print("HaluEval download complete.")
    except Exception as e:
        print(f"Error downloading HaluEval: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download datasets for HalluScope evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["truthfulqa", "halueval"],
        choices=["truthfulqa", "halueval", "all"],
        help="Datasets to download"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["truthfulqa", "halueval"]
    
    print(f"Output directory: {output_dir}")
    print(f"Datasets to download: {datasets}")
    print()
    
    for dataset in datasets:
        if dataset == "truthfulqa":
            download_truthfulqa(output_dir)
        elif dataset == "halueval":
            download_halueval(output_dir)
    
    print("\nDownload complete!")


if __name__ == "__main__":
    main()
