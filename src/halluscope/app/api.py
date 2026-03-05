"""
FastAPI backend for HalluScope V5.

Three-Stage Universal Hallucination Detection:
  Stage 1 — Pre-Hoc Prompt Risk Analysis
  Stage 2 — In-Generation Token Monitoring
  Stage 3 — Post-Generation Semantic Verification

No databases, no retrieval, no external knowledge bases.

Run:
    PYTHONPATH=src uvicorn halluscope.app.api:app --host 0.0.0.0 --port 8000
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
    """Lazy-load the LLM backend V4 on first request."""
    global _llm
    if _llm is None:
        from halluscope.core.llm_backend_v4 import LLMBackend
        _llm = LLMBackend(model_id=MODEL_ID, device="auto")
    return _llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    yield
    global _llm
    _llm = None


app = FastAPI(
    title="HalluScope V5 API",
    description="Three-Stage Universal Hallucination Detection",
    version="5.0.0",
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
    max_new_tokens: int = Field(60, ge=16, le=256, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    hri_threshold: float = Field(0.70, ge=0.0, le=1.0, description="HRI threshold for early stopping")
    early_stop_enabled: bool = Field(True, description="Enable early stopping on high HRI")
    debug: bool = Field(False, description="Enable detailed signal debugging")


class MetricsOut(BaseModel):
    """Uncertainty metrics output."""
    entropy_mean: float = Field(..., description="Mean token entropy (nats)")
    entropy_max: float = Field(..., description="Max token entropy")
    entropy_normalized: float = Field(..., description="Entropy / log(vocab_size)")
    entropy_variance: float = Field(..., description="Variance of token entropies")
    nll: float = Field(..., description="Mean negative log-likelihood")
    perplexity: float = Field(..., description="exp(NLL)")
    logprob_std: float = Field(..., description="Std dev of log-probs")
    logprob_variance: float = Field(..., description="Variance of log-probs")
    semantic_energy: float = Field(..., description="Mean semantic energy")
    kl_uniform_mean: Optional[float] = Field(None, description="KL from uniform (deprecated)")
    num_tokens: int = Field(..., description="Number of generated tokens")


class HRIOut(BaseModel):
    """Hallucination Risk Index output — three-stage blend."""
    score: float = Field(..., ge=0.0, le=1.0, description="Final blended HRI [0,1]")
    label: str = Field(..., description="Risk label: LOW / MEDIUM / HIGH")
    stage2_hri: Optional[float] = Field(None, description="Stage 2 generation HRI")
    verification_risk: Optional[float] = Field(None, description="Stage 3 verification risk")
    prompt_risk: Optional[float] = Field(None, description="Stage 1 prompt risk")


class V4SignalsOut(BaseModel):
    """V4.2 mathematical hallucination detection signals."""
    
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
        description="Mean KL divergence between layer predictions"
    )
    representation_collapse: float = Field(
        ..., 
        description="Similarity between consecutive hidden states"
    )
    
    # V4.2 Semantic Fabrication Signals
    novelty: float = Field(0.0, description="Semantic drift from prompt topic [0,1]")
    entity_expansion: float = Field(0.0, description="Fraction of novel content words [0,1]")
    claim_density: float = Field(0.0, description="Assertions without evidence ratio [0,1]")


class PerTokenSignalsOut(BaseModel):
    """Per-token V4 signal trajectories."""
    hri: List[float] = Field(..., description="HRI after each token")
    csg: List[float] = Field(..., description="CSG at each token")
    lld: List[float] = Field(..., description="LLD at each token")
    rcd: List[float] = Field(..., description="RCD at each token")


class HRIDynamicsOut(BaseModel):
    """HRI rate of change metrics."""
    slope: float = Field(..., description="Rate of HRI change")
    acceleration: float = Field(..., description="Acceleration of HRI change")


class AnalyzeResp(BaseModel):
    """Response schema for /analyze endpoint (V5 three-stage)."""
    answer: str
    hri: HRIOut
    metrics: MetricsOut
    notes: List[str]
    clarification_questions: Optional[List[str]] = None
    
    # Token-level data
    tokens: List[str] = Field(..., description="Generated tokens")
    token_risks: List[float] = Field(..., description="Per-token risk scores [0,1]")
    hri_values: List[float] = Field(..., description="HRI value after each token")
    early_stopped: bool = Field(..., description="True if generation halted early")
    
    # Three-stage breakdown
    stage1_prehoc: Optional[Dict[str, Any]] = Field(None, description="Stage 1 pre-hoc prompt risk")
    stage2_generation: Optional[Dict[str, Any]] = Field(None, description="Stage 2 generation signals")
    stage3_posthoc: Optional[Dict[str, Any]] = Field(None, description="Stage 3 post-hoc verification")
    
    # V4-compatible signals
    v4_signals: V4SignalsOut = Field(..., description="V4 mathematical signals")
    per_token_signals: PerTokenSignalsOut = Field(..., description="Per-token signal trajectories")
    hri_dynamics: HRIDynamicsOut = Field(..., description="HRI rate of change")
    risk_breakdown: Dict[str, float] = Field(default_factory=dict, description="Full risk component breakdown")
    intervention_message: Optional[str] = Field(None, description="Intervention message if triggered")

    # V5.1 Visualization mode fields
    token_diagnostics: Optional[List[Dict[str, Any]]] = Field(
        None, description="Per-token full diagnostics with risk classification"
    )
    hallucination_segments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Consecutive high-risk token segments"
    )
    highlighted_html: Optional[str] = Field(
        None, description="HTML with color-coded tokens by risk"
    )
    highlighted_text: Optional[str] = Field(
        None, description="Plain text with risk markers"
    )
    hri_stats: Optional[Dict[str, float]] = Field(
        None, description="Global HRI statistics (mean, peak, min, final)"
    )
    per_token_entropy: Optional[List[float]] = Field(
        None, description="Entropy values per token"
    )
    per_token_energy: Optional[List[float]] = Field(
        None, description="Semantic energy values per token"
    )
    per_token_logprob: Optional[List[float]] = Field(
        None, description="Log probability values per token"
    )


class HealthResp(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_id: str
    version: str = "5.0.0"
    device: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResp)
def health():
    """
    Health check endpoint.
    
    Returns model status and V4 version information.
    """
    global _llm
    return HealthResp(
        status="ok",
        model_loaded=_llm is not None,
        model_id=MODEL_ID,
        version="5.0.0",
        device=_llm.device if _llm is not None else None,
    )


@app.post("/model/load")
def load_model():
    """
    Explicitly load the model.
    
    Call this before /analyze to ensure the model is ready.
    """
    try:
        llm = get_llm()
        return {
            "status": "loaded",
            "model_id": MODEL_ID,
            "device": llm.device,
            "vocab_size": llm.vocab_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    """
    Analyze a prompt for hallucination risk using V4 mathematical signals.

    Pipeline:
    1. Generate response with per-token V4 signal monitoring
    2. Compute Semantic Energy, Entropy, CSG, LLD, RCD
    3. Maintain running HRI throughout generation
    4. Early stop if HRI exceeds threshold
    5. Return complete signal breakdown

    V4 Signals:
    - **Semantic Energy**: E = -logsumexp(logits). Lower = more confident.
    - **Entropy**: Token uncertainty. Higher = less certain.
    - **CSG**: Confidence * (1 - Stability). Detects confident hallucinations.
    - **LLD**: Layer prediction stability. Higher = unstable reasoning.
    - **RCD**: Hidden state similarity. Higher = repetition/collapse risk.

    Returns:
    - Generated answer with all V4 signals
    - Per-token signal trajectories for visualization
    - HRI score and risk classification
    """
    try:
        from halluscope.app.service_v4 import analyze_prompt
        llm = get_llm()
        result = analyze_prompt(
            llm=llm,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            hri_threshold=req.hri_threshold,
            early_stop_enabled=req.early_stop_enabled,
            debug=req.debug,
        )
        
        # Build V5 response with V5.1 visualization fields
        return AnalyzeResp(
            answer=result["answer"],
            hri=HRIOut(**result["hri"]),
            metrics=MetricsOut(**result["metrics"]),
            notes=result["notes"],
            clarification_questions=result.get("clarification_questions"),
            tokens=result["tokens"],
            token_risks=result["token_risks"],
            hri_values=result["hri_values"],
            early_stopped=result["early_stopped"],
            stage1_prehoc=result.get("stage1_prehoc"),
            stage2_generation=result.get("stage2_generation"),
            stage3_posthoc=result.get("stage3_posthoc"),
            v4_signals=V4SignalsOut(**result["v4_signals"]),
            per_token_signals=PerTokenSignalsOut(**result["per_token_signals"]),
            hri_dynamics=HRIDynamicsOut(**result["hri_dynamics"]),
            risk_breakdown=result.get("risk_breakdown", {}),
            intervention_message=result.get("intervention_message"),
            # V5.1 visualization fields
            token_diagnostics=result.get("token_diagnostics"),
            hallucination_segments=result.get("hallucination_segments"),
            highlighted_html=result.get("highlighted_html"),
            highlighted_text=result.get("highlighted_text"),
            hri_stats=result.get("hri_stats"),
            per_token_entropy=result.get("per_token_entropy"),
            per_token_energy=result.get("per_token_energy"),
            per_token_logprob=result.get("per_token_logprob"),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick_analyze")
def quick_analyze(req: AnalyzeReq) -> Dict[str, Any]:
    """
    Quick analysis returning simplified JSON.
    
    Use for rapid testing or when you need raw signal access.
    """
    try:
        llm = get_llm()
        result = llm.analyze_generation(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            debug=req.debug,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/info")
def signal_info():
    """
    Get information about V4 hallucination detection signals.
    """
    return {
        "version": "4.0.0",
        "signals": {
            "semantic_energy": {
                "formula": "E = -logsumexp(logits)",
                "description": "Measures confidence intensity. Lower = more confident.",
                "range": "[0, 1] (normalized)",
            },
            "entropy": {
                "formula": "H = -Σ p * log(p)",
                "description": "Token prediction uncertainty.",
                "range": "[0, 1] (normalized)",
            },
            "confidence_stability_gap": {
                "formula": "CSG = confidence * (1 - semantic_stability)",
                "description": "Detects confident hallucinations.",
                "range": "[0, 1]",
            },
            "logit_lens_divergence": {
                "formula": "LLD = mean(KL(P_i || P_{i+1}))",
                "description": "Prediction stability across layers.",
                "range": "[0, 1] (normalized)",
            },
            "representation_collapse": {
                "formula": "RCD = cosine(h_t, h_{t-1})",
                "description": "Hidden state repetition detection.",
                "range": "[0, 1]",
            },
            "semantic_novelty": {
                "formula": "novelty = 1 - cosine(prompt_emb, answer_emb)",
                "description": "Semantic drift from prompt. Catches confident fabrication.",
                "range": "[0, 1]",
            },
            "entity_expansion": {
                "formula": "entity_exp = new_content_words / total_content_words",
                "description": "Novel vocabulary ratio. High = inventing new concepts.",
                "range": "[0, 1]",
            },
            "claim_density": {
                "formula": "claim = assertions / (assertions + evidence + 1)",
                "description": "Ungrounded assertion ratio. High = confident claims without evidence.",
                "range": "[0, 1]",
            },
        },
        "hri_formula": "HRI = sigmoid(\u03a3 w_i * signal_i * 10 + bias)",
        "weights": {
            "w_entropy": 0.10,
            "w_energy": 0.05,
            "w_variance": 0.10,
            "w_csg": 0.10,
            "w_lld": 0.10,
            "w_rcd": 0.05,
            "w_novelty": 0.25,
            "w_entity": 0.10,
            "w_claim": 0.15,
            "bias": -1.5,
        },
    }
