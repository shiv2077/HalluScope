"""
HalluScope – Legacy API Routes (/analyze)

Kept for backward compatibility with v1 frontend.
"""

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from backend.app.adapters.openai_adapter import OpenAIAdapter
from backend.app.services.analysis_service import analyze_prompt
from backend.app.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

router = APIRouter()


class AnalyzeRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)


class LegacyMetricsResponse(BaseModel):
    mean_logprob: float
    logprob_std: float
    confidence_decay: float


class AnalyzeResponse(BaseModel):
    answer: str
    hri: float
    risk_level: str
    metrics: LegacyMetricsResponse
    model: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
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
