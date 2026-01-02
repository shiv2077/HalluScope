"""
Pydantic Schemas Module.

This module defines request and response schemas for the
HalluScope API using Pydantic models.
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


class MetricsDetail(BaseModel):
    """Detailed metrics breakdown."""
    
    entropy: float = Field(..., description="Entropy-based uncertainty")
    logprob: float = Field(..., description="Log-probability confidence")
    disagreement: float = Field(..., description="Multi-sample disagreement")
    edfl_proxy: Optional[float] = Field(None, description="EDFL proxy score")


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    
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
