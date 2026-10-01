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
    # Optional reproducibility controls. temperature=0 => greedy/deterministic
    # decoding; seed seeds all RNGs before generation. Used by the benchmark
    # runner's --eval-deterministic path; None => server defaults.
    temperature: Optional[float] = None
    seed: Optional[int] = None
    # Live serving suppresses high-risk answers with a clarification. The
    # benchmark sets this False so the REAL model answer is returned for
    # grading (the intervention decision is still reported). Default True keeps
    # the deployed chat UI's protective behaviour unchanged.
    apply_intervention: bool = True


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
    # ── Local-mode internal signals (None in API mode) ──────────────
    mean_logit_flatness: Optional[float] = None
    # Deprecated alias of mean_logit_flatness, kept so existing frontend code
    # reading the old key does not break. Same value; do not diverge them.
    mean_semantic_energy: Optional[float] = None
    mean_rcd: Optional[float] = None
    min_rcd: Optional[float] = None
    mean_lld: Optional[float] = None  # Logit Lens Divergence (needs per-layer)
    mean_csg: Optional[float] = None  # Confidence-Stability Gap
    hri_mode: str = "api"  # "api" | "local_internal"


class InternalPerToken(BaseModel):
    """Per-token traces of internal signals (local mode only)."""

    energy: List[float] = []
    rcd: List[float] = []


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
    internal_per_token: Optional[InternalPerToken] = None
    # Stage 1 (s1..s4 + aggregate) and Stage 3 (novelty/expansion/density).
    prompt_risk: Optional[dict] = None
    semantic_checks: Optional[dict] = None


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
    intervened: bool = False
    internal_per_token: Optional[InternalPerToken] = None
    prompt_risk: Optional[dict] = None
    semantic_checks: Optional[dict] = None
