#!/bin/bash
# Run API Server Script for HalluScope
#
# This script starts the FastAPI backend server.

set -e

# Default values
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
RELOAD="${API_RELOAD:-true}"

echo "=== Starting HalluScope API Server ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Reload: $RELOAD"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to include src
export PYTHONPATH="src:${PYTHONPATH}"

# Run the server
if [ "$RELOAD" = "true" ]; then
    uvicorn halluscope.app.api:app --host "$HOST" --port "$PORT" --reload
else
    uvicorn halluscope.app.api:app --host "$HOST" --port "$PORT"
fi
