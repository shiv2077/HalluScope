"""
HalluScope - Configuration Module

Loads environment variables and provides centralized configuration
for the backend application. API keys and model defaults live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the monorepo root (one level above backend/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# ── API Keys ─────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Model Defaults ───────────────────────────────────────────────────
DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL: str = os.getenv(
    "LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
DEFAULT_MAX_TOKENS: int = 80
DEFAULT_TEMPERATURE: float = 0.7
