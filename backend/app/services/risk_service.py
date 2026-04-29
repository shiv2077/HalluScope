"""
HalluScope v2 – Risk Service

Orchestrates the full analysis: LLM call → metric computation → HRI.
Returns everything needed for the chat response and metrics page.
"""

from backend.app.services.llm_service import LLMService
from backend.app.risk.hri_engine import compute_hri
from backend.app.models.schemas import MetricsDetail, TokenDetail, InternalPerToken
from backend.app.metrics.internal_metrics import aggregate_internal_signals


class RiskService:
    """Stateful service — wraps an LLMService and computes risk
    for every assistant reply."""

    def __init__(self, evaluation_mode: str = "api"):
        self.evaluation_mode = evaluation_mode
        self.llm = LLMService(evaluation_mode=evaluation_mode)

    def analyze_message(self, user_message: str) -> dict:
        """Run the full pipeline for a user message.

        Returns a dict with keys: text, hri, risk_level, metrics,
        token_details, per_token_hri, internal_per_token, model.
        """

        result = self.llm.send_message(user_message)

        # ── Compute internal signals when the adapter exposed them ──
        # API adapters leave hidden_states_per_token / logits_per_token as
        # None, so aggregate_internal_signals short-circuits to all-Nones
        # and compute_hri falls back to the API formula.
        internal = aggregate_internal_signals(
            result.logits_per_token,
            result.hidden_states_per_token,
        )

        hri_result = compute_hri(
            logprobs=result.token_logprobs,
            top_logprobs=result.top_logprobs,
            internal_signals=internal,
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

        # Merge internal-signal summaries into MetricsDetail
        metrics_payload = dict(hri_result["metrics"])
        metrics_payload["mean_semantic_energy"] = internal.get("mean_semantic_energy")
        metrics_payload["mean_rcd"] = internal.get("mean_rcd")
        metrics_payload["min_rcd"] = internal.get("min_rcd")
        metrics_payload["hri_mode"] = hri_result["hri_mode"]
        metrics = MetricsDetail(**metrics_payload)

        # Per-token internal trajectories — only meaningful when we
        # actually had internal access; else leave None so the frontend
        # knows not to render the local-only chart.
        if internal.get("mean_semantic_energy") is not None:
            internal_per_token = InternalPerToken(
                energy=internal.get("per_token_energy", []),
                rcd=internal.get("per_token_rcd", []),
            )
        else:
            internal_per_token = None

        return {
            "text": result.text,
            "hri": hri_result["hri"],
            "risk_level": hri_result["risk_level"],
            "metrics": metrics,
            "token_details": token_details,
            "per_token_hri": hri_result["per_token_hri"],
            "internal_per_token": internal_per_token,
            "model": result.model_name,
            "evaluation_mode": self.evaluation_mode,
        }

    def inject_assistant_message(self, content: str):
        """Manually add an assistant message to conversation history."""
        self.llm.add_assistant_message(content)

    def pop_last_assistant(self) -> None:
        """Strip the most recent assistant message from history."""
        self.llm.pop_last_assistant()
