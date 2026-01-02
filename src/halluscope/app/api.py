"""
FastAPI backend for HalluScope.

Run:
    PYTHONPATH=src uvicorn halluscope.app.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lazy import to avoid loading model at module import time
_llm = None


def get_llm():
    """Lazy-load the LLM backend on first request."""
    global _llm
    if _llm is None:
        from halluscope.core.llm_backend import LLMBackend
        _llm = LLMBackend(model_id="Qwen/Qwen2.5-3B-Instruct", device="auto")
    return _llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: optionally pre-load model (comment out for lazy loading)
    # get_llm()
    yield
    # Shutdown: cleanup if needed
    global _llm
    _llm = None


app = FastAPI(
    title="HalluScope API",
    description="Proactive Hallucination Risk Estimation for LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeReq(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192, description="User prompt")
    max_new_tokens: int = Field(256, ge=16, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)


class MetricsOut(BaseModel):
    entropy_mean: float = Field(..., description="Mean token entropy (nats)")
    nll: float = Field(..., description="Mean negative log-likelihood")
    # Sprint-2 placeholders
    disagreement: Optional[float] = Field(None, description="(Sprint-2) Multi-sample disagreement")
    edfl_proxy: Optional[float] = Field(None, description="(Sprint-3) EDFL evidence sensitivity")


class AnalyzeResp(BaseModel):
    answer: str
    hri: float = Field(..., ge=0.0, le=1.0, description="Hallucination Risk Index")
    risk_level: str = Field(..., description="low / medium / high")
    metrics: MetricsOut
    notes: List[str]
    clarification_questions: Optional[List[str]] = None


class HealthResp(BaseModel):
    status: str
    model_loaded: bool
    model_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResp)
def health():
    """Health check endpoint."""
    return HealthResp(
        status="ok",
        model_loaded=_llm is not None,
        model_id="Qwen/Qwen2.5-3B-Instruct",
    )


@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    """
    Analyze a prompt for hallucination risk.

    Returns the generated answer along with HRI score, uncertainty metrics,
    explanation notes, and clarification questions if risk is high.
    """
    try:
        from halluscope.app.service import analyze_prompt
        llm = get_llm()
        result = analyze_prompt(
            llm=llm,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
