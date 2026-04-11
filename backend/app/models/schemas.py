"""
HalluScope v2 – Pydantic Schemas

Defines all request/response models used across the API.
Centralised here so that API routes, services, and tests
can import from a single source of truth.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ── Chat ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /chat body."""
    message: str
    evaluation_mode: Literal["api", "local"] = "api"


class TokenDetail(BaseModel):
    """Per-token uncertainty detail returned with each response."""
    token: str
    logprob: float
    entropy: float = 0.0
    risk_contribution: float = 0.0


class MetricsDetail(BaseModel):
    """Extended uncertainty metrics for a single response."""
    mean_logprob: float = 0.0
    logprob_std: float = 0.0
    logprob_variance: float = 0.0
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    neg_log_likelihood: float = 0.0
    perplexity: float = 0.0
    confidence_decay: float = 0.0
    token_count: int = 0


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str  # "user" | "assistant"
    content: str
    risk_level: Optional[str] = None
    hri_score: Optional[float] = None
    metrics: Optional[MetricsDetail] = None
    token_details: Optional[List[TokenDetail]] = None
    intervention: bool = False


class ChatResponse(BaseModel):
    """POST /chat response."""
    answer: str
    hri: float
    risk_level: str
    metrics: MetricsDetail
    token_details: List[TokenDetail] = []
    intervention: bool = False
    model: str = ""
    evaluation_mode: Literal["api", "local"] = "api"
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


# ── Metrics lookup ───────────────────────────────────────────────────

class MetricsLookupResponse(BaseModel):
    """GET /metrics/{message_id} response."""
    message_id: str
    content: str
    hri: float
    risk_level: str
    metrics: MetricsDetail
    token_details: List[TokenDetail] = []
    per_token_hri: List[float] = []
    model: str = ""
    evaluation_mode: Literal["api", "local"] = "api"
