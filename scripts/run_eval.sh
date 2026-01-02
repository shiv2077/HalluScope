#!/bin/bash
# Run Evaluation Script for HalluScope
#
# This script runs the hallucination detection evaluation pipeline.

set -e

# Default values
DATASET="${EVAL_DATASET:-truthfulqa}"
MODEL="${EVAL_MODEL:-meta-llama/Llama-2-7b-hf}"
OUTPUT_DIR="${EVAL_OUTPUT:-outputs/eval}"

echo "=== Running HalluScope Evaluation ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to include src
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python -m halluscope.eval.run_eval \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_DIR"
