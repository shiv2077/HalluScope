"""
HalluScope v2 – Application Entry Point

Creates and configures the FastAPI application with CORS middleware.
Registers both the legacy /analyze endpoint and the new v2 chat +
metrics routers.

Run with:
    uvicorn backend.app.main:app --reload
"""

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import router as legacy_router  # /analyze
from backend.app.api.chat import router as chat_router  # /chat
from backend.app.api.metrics import router as metrics_router  # /metrics
from backend.app import storage
from backend.app.config import DEFAULT_LOCAL_MODEL

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


# ── Startup hooks ────────────────────────────────────────────────────
def _warm_local_model() -> None:
    """Load the local HuggingFace model in a background thread.

    Importing transformers/torch and downloading weights can take 30-300s
    on a cold cache. Doing it in the request handler blocks the event loop
    and the user sees a forever-spinner. Doing it on the startup thread
    blocks /health. Daemon thread keeps both responsive.
    """
    try:
        from backend.app.adapters.local_adapter import LocalAdapter

        LocalAdapter._ensure_model_loaded()
    except Exception:
        # Don't crash the server if local deps are missing; /health will
        # continue to report local_model_loaded=false.
        import traceback

        traceback.print_exc()


@app.on_event("startup")
def _on_startup() -> None:
    storage.init_db()
    threading.Thread(target=_warm_local_model, daemon=True).start()


# ── Health endpoint ──────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    """Liveness + local-model readiness."""
    loaded = False
    try:
        from backend.app.adapters.local_adapter import LocalAdapter

        loaded = LocalAdapter._model is not None
    except Exception:
        loaded = False
    return {
        "status": "ok",
        "local_model_loaded": loaded,
        "local_model_name": DEFAULT_LOCAL_MODEL,
    }


# ── Register routes ──────────────────────────────────────────────────
app.include_router(legacy_router)  # backward compat: POST /analyze
app.include_router(chat_router)  # v2: POST /chat, POST /chat/reset
app.include_router(metrics_router)  # v2: GET /metrics/{message_id}
