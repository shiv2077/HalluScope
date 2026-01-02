#!/bin/bash
# Setup Environment Script for HalluScope
#
# This script sets up the development environment including
# virtual environment creation and dependency installation.

set -e

echo "=== HalluScope Environment Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing HalluScope in development mode..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed data/samples
mkdir -p logs
mkdir -p outputs/eval

# Copy environment file if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env with your configuration."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the API:"
echo "  uvicorn halluscope.app.api:app --reload"
echo ""
echo "To run the UI:"
echo "  streamlit run src/halluscope/app/ui.py"
