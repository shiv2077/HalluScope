"""
HalluScope v2 – Metrics Endpoint

GET /metrics/{message_id}
    Returns full uncertainty analysis for a specific message,
    including per-token HRI trajectory and token details.
"""

from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import MetricsLookupResponse
from backend.app import storage

router = APIRouter()


@router.get("/metrics/{message_id}", response_model=MetricsLookupResponse)
async def get_metrics(message_id: str):
    """Look up detailed metrics for a previously analysed message."""
    entry = storage.get_message(message_id)

    if not entry:
        raise HTTPException(status_code=404, detail="Message not found")

    return MetricsLookupResponse(
        message_id=message_id,
        content=entry["content"],
        hri=entry["hri"],
        risk_level=entry["risk_level"],
        metrics=entry["metrics"],
        token_details=entry["token_details"],
        per_token_hri=entry["per_token_hri"],
        model=entry["model"],
        evaluation_mode=entry.get("evaluation_mode", "api"),
        intervened=entry.get("intervened", False),
        internal_per_token=entry.get("internal_per_token"),
        prompt_risk=entry.get("prompt_risk"),
        semantic_checks=entry.get("semantic_checks"),
    )
