"""
FastAPI backend for HalluScope.

TokenBurn-inspired hallucination risk estimation API.

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
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def get_llm():
    """Lazy-load the LLM backend on first request."""
    global _llm
    if _llm is None:
        from halluscope.core.llm_backend import LLMBackend
        _llm = LLMBackend(model_id=MODEL_ID, device="auto")
    return _llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    yield
    global _llm
    _llm = None


app = FastAPI(
    title="HalluScope API",
    description="TokenBurn-inspired Proactive Hallucination Risk Estimation for LLMs",
    version="1.1.0",
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
    """Request schema for /analyze endpoint."""
    prompt: str = Field(..., min_length=1, max_length=8192, description="User prompt")
    max_new_tokens: int = Field(256, ge=16, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    compute_kl: bool = Field(True, description="Compute KL divergence from uniform")


class MetricsOut(BaseModel):
    """TokenBurn-inspired uncertainty metrics."""
    entropy_mean: float = Field(..., description="Mean token entropy (nats)")
    entropy_max: float = Field(..., description="Max token entropy")
    entropy_normalized: float = Field(..., description="Entropy / log(vocab_size)")
    entropy_variance: float = Field(..., description="Variance of token entropies")
    nll: float = Field(..., description="Mean negative log-likelihood")
    perplexity: float = Field(..., description="exp(NLL) - interpretable confidence")
    logprob_std: float = Field(..., description="Std dev of log-probs (instability)")
    logprob_variance: float = Field(..., description="Variance of log-probs")
    kl_uniform_mean: Optional[float] = Field(None, description="KL divergence from uniform")
    num_tokens: int = Field(..., description="Number of generated tokens")


class HRIOut(BaseModel):
    """Hallucination Risk Index output."""
    score: float = Field(..., ge=0.0, le=1.0, description="HRI score [0,1]")
    label: str = Field(..., description="Risk label: LOW / MEDIUM / HIGH")


class AnalyzeResp(BaseModel):
    """Response schema for /analyze endpoint."""
    answer: str
    hri: HRIOut
    metrics: MetricsOut
    notes: List[str]
    clarification_questions: Optional[List[str]] = None


class HealthResp(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_id: str
    device: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResp)
def health():
    """
    Health check endpoint.
    
    Returns model status, ID, and device information.
    """
    global _llm
    return HealthResp(
        status="ok",
        model_loaded=_llm is not None,
        model_id=MODEL_ID,
        device=_llm.device if _llm is not None else None,
    )


@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    """
    Analyze a prompt for hallucination risk.

    TokenBurn-inspired pipeline:
    1. Generate response with per-step logits
    2. Compute entropy, NLL, perplexity, variance, KL
    3. Compute HRI score and risk label
    4. Return with explanation notes and clarification questions

    Returns the generated answer along with HRI score/label, uncertainty metrics,
    explanation notes, and clarification questions if risk is elevated.
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
            compute_kl=req.compute_kl,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
