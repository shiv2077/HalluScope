"""
HalluScope – API Routes

Defines the REST endpoints exposed by the backend.
Currently provides a single route:

    POST /analyze   – accept a prompt and return the hallucination
                      risk analysis.
"""

from typing import Literal

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from backend.app.adapters.openai_adapter import OpenAIAdapter
from backend.app.services.analysis_service import analyze_prompt
from backend.app.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

router = APIRouter()


# ── Request / Response schemas ───────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """JSON body accepted by POST /analyze."""
    prompt: str
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    evaluation_mode: Literal["api", "local"] = "api"


class MetricsResponse(BaseModel):
    mean_logprob: float
    logprob_std: float
    confidence_decay: float


class AnalyzeResponse(BaseModel):
    """JSON body returned by POST /analyze."""
    answer: str
    hri: float
    risk_level: str
    metrics: MetricsResponse
    model: str


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyse a prompt for hallucination risk.

    1. Forwards the prompt to the configured LLM adapter.
    2. Computes token-level metrics and Hallucination Risk Index.
    3. Returns the generated answer alongside risk information.
    """
    try:
        if request.evaluation_mode == "local":
            from backend.app.adapters.local_adapter import LocalAdapter

            adapter = LocalAdapter()
        else:
            adapter = OpenAIAdapter()
        result = analyze_prompt(
            adapter=adapter,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
