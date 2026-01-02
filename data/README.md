# Data Directory

This directory contains datasets for HalluScope.

## Structure

```
data/
├── raw/           # Original, unprocessed datasets
├── processed/     # Cleaned and preprocessed data
├── samples/       # Sample data for testing and demos
└── README.md      # This file
```

## Datasets

### Supported Benchmarks

1. **TruthfulQA** - Questions designed to test truthfulness
2. **HaluEval** - Hallucination evaluation benchmark

### Downloading Data

Run the download script:

```bash
python scripts/download_datasets.py
```

## Note

Large dataset files are ignored by git. Use the download script to fetch them.
