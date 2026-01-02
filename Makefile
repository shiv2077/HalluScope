# HalluScope Makefile

.PHONY: install dev-install test lint format api ui eval clean help

# Default target
help:
	@echo "HalluScope - Available Commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make api          - Run FastAPI server"
	@echo "  make ui           - Run Streamlit UI"
	@echo "  make eval         - Run evaluation pipeline"
	@echo "  make clean        - Clean build artifacts"

# Install dependencies
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/halluscope --cov-report=html

# Linting and formatting
lint:
	flake8 src/halluscope tests
	mypy src/halluscope

format:
	black src/halluscope tests
	isort src/halluscope tests

# Run applications
api:
	uvicorn halluscope.app.api:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run src/halluscope/app/ui.py

# Evaluation
eval:
	python -m halluscope.eval.run_eval

# Download datasets
download-data:
	python scripts/download_datasets.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
