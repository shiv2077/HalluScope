# HalluScope V4 Makefile

.PHONY: install test api ui clean help

help:
	@echo "HalluScope V4 — Commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make api      - Start FastAPI server (port 8000)"
	@echo "  make ui       - Start Streamlit UI (port 8501)"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean caches"

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest tests/ -v

api:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src uvicorn halluscope.app.api:app --reload --host 0.0.0.0 --port 8000

ui:
	PYTHONPATH=src streamlit run src/halluscope/app/ui.py --server.port 8501

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
