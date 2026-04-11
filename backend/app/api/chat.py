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

router = APIRouter()

# In-memory risk services by evaluation mode (maintain independent history).
# In production this would be per-session / per-user.
_risk_services: dict[str, RiskService] = {
    "api": RiskService(evaluation_mode="api"),
    "local": RiskService(evaluation_mode="local"),
}

# In-memory store for metrics lookup by message_id
_message_store: dict = {}


def get_message_store() -> dict:
    """Expose store so the metrics endpoint can read it."""
    return _message_store


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a single chat turn with hallucination monitoring."""
    try:
        mode = request.evaluation_mode
        risk_service = _risk_services[mode]
        analysis = risk_service.analyze_message(request.message)

        message_id = uuid.uuid4().hex[:12]
        intervention = should_intervene(analysis["hri"])

        if intervention:
            # Suppress the LLM answer and return clarification instead
            clarification = generate_clarification(request.message)
            risk_service.inject_assistant_message(clarification)

            response = ChatResponse(
                answer=clarification,
                hri=analysis["hri"],
                risk_level=analysis["risk_level"],
                metrics=analysis["metrics"],
                token_details=analysis["token_details"],
                intervention=True,
                model=analysis["model"],
                evaluation_mode=analysis["evaluation_mode"],
                message_id=message_id,
            )
        else:
            response = ChatResponse(
                answer=analysis["text"],
                hri=analysis["hri"],
                risk_level=analysis["risk_level"],
                metrics=analysis["metrics"],
                token_details=analysis["token_details"],
                intervention=False,
                model=analysis["model"],
                evaluation_mode=analysis["evaluation_mode"],
                message_id=message_id,
            )

        # Store for metrics page lookup
        _message_store[message_id] = {
            "content": analysis["text"],
            "hri": analysis["hri"],
            "risk_level": analysis["risk_level"],
            "metrics": analysis["metrics"],
            "token_details": analysis["token_details"],
            "per_token_hri": analysis["per_token_hri"],
            "model": analysis["model"],
            "evaluation_mode": analysis["evaluation_mode"],
        }

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
