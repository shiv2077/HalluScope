#!/bin/bash
# Run Streamlit UI Script for HalluScope
#
# This script starts the Streamlit web interface.

set -e

# Default values
PORT="${UI_PORT:-8501}"

echo "=== Starting HalluScope UI ==="
echo "Port: $PORT"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to include src
export PYTHONPATH="src:${PYTHONPATH}"

# Run Streamlit
streamlit run src/halluscope/app/ui.py --server.port "$PORT"
