"""
Pydantic Schemas Module - HalluScope V4.

This module defines request and response schemas for the
HalluScope V4 API using Pydantic models.

V4 uses purely mathematical signals:
- Semantic Energy
- Entropy / Logprob Variance
- Confidence-Stability Gap (CSG)
- Logit Lens Divergence (LLD)
- Representation Collapse Detection (RCD)
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    
    query: str = Field(
        ...,
        description="The input query or question to analyze",
        min_length=1,
        max_length=4096,
    )
    context: Optional[str] = Field(
        None,
        description="Optional context for the query",
        max_length=8192,
    )
    num_samples: int = Field(
        5,
        description="Number of samples for disagreement metrics",
        ge=1,
        le=20,
    )
    temperature: float = Field(
        0.7,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        256,
        description="Maximum tokens to generate",
        ge=1,
        le=2048,
    )


# ─────────────────────────────────────────────────────────────────────────────
# V4 Signal Schemas
# ─────────────────────────────────────────────────────────────────────────────

class V4SignalsOut(BaseModel):
    """V4 hallucination detection signals (purely mathematical)."""
    
    # Distribution Geometry
    entropy: float = Field(..., description="Normalized token entropy [0,1]")
    semantic_energy: float = Field(..., description="Semantic energy E=-logsumexp [0,1]")
    logprob_variance: float = Field(..., description="Variance of log probabilities")
    
    # Confidence-Stability Gap
    confidence: float = Field(..., description="Model confidence sigmoid(-mean_logprob)")
    semantic_stability: float = Field(..., description="Cosine similarity to prompt embedding")
    confidence_stability_gap: float = Field(
        ..., 
        description="CSG = confidence * (1 - stability). High = confident hallucination"
    )
    
    # Internal Representation Analysis
    logit_lens_divergence: float = Field(
        ..., 
        description="LLD: Mean KL divergence between layer predictions"
    )
    representation_collapse: float = Field(
        ..., 
        description="RCD: Cosine similarity between consecutive hidden states"
    )


class PerTokenSignalsOut(BaseModel):
    """Per-token V4 signal trajectories."""
    
    hri: List[float] = Field(..., description="HRI after each token")
    csg: List[float] = Field(..., description="CSG at each token")
    lld: List[float] = Field(..., description="LLD at each token")
    rcd: List[float] = Field(..., description="RCD at each token")


class MetricsDetail(BaseModel):
    """Detailed metrics breakdown - V4 compatible."""
    
    entropy: float = Field(..., description="Entropy-based uncertainty")
    semantic_energy: float = Field(..., description="Semantic energy score")
    logprob_variance: float = Field(..., description="Logprob variance (instability)")
    confidence: float = Field(..., description="Model confidence")
    semantic_stability: float = Field(..., description="Semantic stability score")
    edfl_proxy: Optional[float] = Field(None, description="EDFL proxy score (deprecated)")


class QueryResponse(BaseModel):
    """Response schema for query endpoint - V4 compatible."""
    
    response: str = Field(..., description="Generated response text")
    hri_score: float = Field(
        ...,
        description="Hallucination Risk Index (0-1)",
        ge=0.0,
        le=1.0,
    )
    risk_level: str = Field(
        ...,
        description="Risk level category",
    )
    metrics: Dict[str, float] = Field(
        ...,
        description="Individual metric scores",
    )
    v4_signals: Optional[V4SignalsOut] = Field(
        None,
        description="V4 mathematical hallucination signals",
    )
    intervention: Optional[str] = Field(
        None,
        description="Intervention message if risk is elevated",
    )
    samples: Optional[List[str]] = Field(
        None,
        description="Generated samples (if requested)",
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class BatchQueryRequest(BaseModel):
    """Request schema for batch processing."""
    
    queries: List[QueryRequest] = Field(
        ...,
        description="List of queries to process",
        min_items=1,
        max_items=100,
    )


class BatchQueryResponse(BaseModel):
    """Response schema for batch processing."""
    
    responses: List[QueryResponse] = Field(
        ...,
        description="List of responses",
    )
    total: int = Field(..., description="Total queries processed")
    failed: int = Field(0, description="Number of failed queries")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    code: str = Field(..., description="Error code")
