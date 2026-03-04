"""
HalluScope v2 – Application Entry Point

Creates and configures the FastAPI application with CORS middleware.
Registers both the legacy /analyze endpoint and the new v2 chat +
metrics routers.

Run with:
    uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router as legacy_router          # /analyze
from backend.app.api.chat import router as chat_router        # /chat
from backend.app.api.metrics import router as metrics_router  # /metrics

# ── App factory ──────────────────────────────────────────────────────
app = FastAPI(
    title="HalluScope API",
    description="Reliability layer for LLMs – analyses token-level "
                "uncertainty signals to estimate hallucination risk.",
    version="0.2.0",
)

# Allow the Vite dev-server origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ──────────────────────────────────────────────────
app.include_router(legacy_router)    # backward compat: POST /analyze
app.include_router(chat_router)      # v2: POST /chat, POST /chat/reset
app.include_router(metrics_router)   # v2: GET /metrics/{message_id}
