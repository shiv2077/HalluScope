"""
HalluScope v2 – Chat Endpoint

POST /chat
    Accepts a user message, runs it through the LLM + risk pipeline,
    applies intervention logic, and returns the structured response.

The RiskService instance is shared across requests so that
conversation history is maintained for the session.
"""

import uuid
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
)
from backend.app.services.risk_service import RiskService
from backend.app.services.intervention_service import (
    should_intervene,
    generate_clarification,
)
from backend.app import storage

router = APIRouter()

# In-memory risk services by evaluation mode (maintain independent history).
# In production this would be per-session / per-user.
_risk_services: dict[str, RiskService] = {
    "api": RiskService(evaluation_mode="api"),
    "local": RiskService(evaluation_mode="local"),
}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a single chat turn with hallucination monitoring."""
    try:
        mode = request.evaluation_mode
        risk_service = _risk_services[mode]

        # Reproducibility: seed all RNGs before generation when requested
        # (used by the benchmark's --eval-deterministic path).
        if request.seed is not None:
            from backend.app.seeding import set_global_seed
            set_global_seed(request.seed)

        analysis = risk_service.analyze_message(
            request.message, temperature=request.temperature
        )

        message_id = uuid.uuid4().hex[:12]
        intervention = should_intervene(analysis["hri"])

        if intervention:
            # Suppress the LLM answer and return clarification instead.
            # send_message() already appended the real reply to history; strip
            # it before injecting the clarification so future turns don't see
            # the suppressed answer in context.
            clarification = generate_clarification(request.message)
            risk_service.pop_last_assistant()
            risk_service.inject_assistant_message(clarification)
            displayed_content = clarification
        else:
            displayed_content = analysis["text"]

        response = ChatResponse(
            answer=displayed_content,
            hri=analysis["hri"],
            risk_level=analysis["risk_level"],
            metrics=analysis["metrics"],
            token_details=analysis["token_details"],
            intervention=intervention,
            model=analysis["model"],
            evaluation_mode=analysis["evaluation_mode"],
            message_id=message_id,
            internal_per_token=analysis.get("internal_per_token"),
        )

        # Persist for metrics page lookup. When intervention fires we
        # deliberately store the clarification (not the suppressed LLM text)
        # so the metrics page cannot leak the answer the user was protected
        # from.
        storage.save_message(message_id, {
            "content": displayed_content,
            "hri": analysis["hri"],
            "risk_level": analysis["risk_level"],
            "metrics": analysis["metrics"],
            "token_details": analysis["token_details"],
            "per_token_hri": analysis["per_token_hri"],
            "model": analysis["model"],
            "evaluation_mode": analysis["evaluation_mode"],
            "intervened": intervention,
            "internal_per_token": analysis.get("internal_per_token"),
        })

        return response

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/reset")
async def reset_chat():
    """Reset conversation history."""
    global _risk_services
    _risk_services = {
        "api": RiskService(evaluation_mode="api"),
        "local": RiskService(evaluation_mode="local"),
    }
    return {"status": "ok"}
