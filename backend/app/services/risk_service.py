"""
HalluScope v2 – Risk Service

Orchestrates the full analysis: LLM call → metric computation → HRI.
Returns everything needed for the chat response and metrics page.
"""

from backend.app.services.llm_service import LLMService
from backend.app.risk.hri_engine import compute_hri
from backend.app.models.schemas import MetricsDetail, TokenDetail


class RiskService:
    """Stateful service — wraps an LLMService and computes risk
    for every assistant reply."""

    def __init__(self):
        self.llm = LLMService()

    def analyze_message(self, user_message: str) -> dict:
        """Run the full pipeline for a user message.

        Returns a dict with keys: text, hri, risk_level, metrics,
        token_details, per_token_hri, model.
        """

        result = self.llm.send_message(user_message)

        hri_result = compute_hri(
            logprobs=result.token_logprobs,
            top_logprobs=result.top_logprobs,
        )

        # Build per-token details for heatmap
        token_details = []
        for i, lp in enumerate(result.token_logprobs):
            tok_str = result.token_strings[i] if i < len(result.token_strings) else ""
            ent = hri_result["token_entropies"][i] if i < len(hri_result["token_entropies"]) else 0.0
            rc = hri_result["risk_contributions"][i] if i < len(hri_result["risk_contributions"]) else 0.0
            token_details.append(TokenDetail(
                token=tok_str,
                logprob=round(lp, 4),
                entropy=round(ent, 4),
                risk_contribution=round(rc, 4),
            ))

        metrics = MetricsDetail(**hri_result["metrics"])

        return {
            "text": result.text,
            "hri": hri_result["hri"],
            "risk_level": hri_result["risk_level"],
            "metrics": metrics,
            "token_details": token_details,
            "per_token_hri": hri_result["per_token_hri"],
            "model": result.model_name,
        }

    def inject_assistant_message(self, content: str):
        """Manually add an assistant message to conversation history."""
        self.llm.add_assistant_message(content)
