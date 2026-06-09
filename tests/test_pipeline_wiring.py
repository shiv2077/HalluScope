"""Integration: Stage 1 (pre-gen) and Stage 3 (post-gen) signals must appear in
the RiskService output. Uses a stub LLM so no model/API is needed; novelty is
disabled via env so SBERT is not loaded.
"""

from __future__ import annotations


import pytest

from backend.app.adapters.schemas import StandardizedResult


class _StubLLM:
    """Stands in for LLMService: returns a fixed API-style result."""

    def __init__(self):
        self.history = []

    def send_message(self, user_message, temperature=None):
        return StandardizedResult(
            text="Zbigniew Brzezinski was born in Warsaw in 1928.",
            token_logprobs=[-0.5, -0.6, -0.4, -0.7, -0.55, -0.6],
            token_strings=["Z", "big", "niew", " was", " born", "."],
            top_logprobs=[[-0.5, -1.0, -2.0, -3.0, -4.0]] * 6,
            model_name="stub",
        )

    def add_assistant_message(self, content):
        self.history.append(content)

    def pop_last_assistant(self):
        pass


@pytest.fixture(autouse=True)
def _disable_novelty(monkeypatch):
    monkeypatch.setenv("HALLUSCOPE_ENABLE_NOVELTY", "0")


def test_analyze_message_includes_prompt_risk_and_semantic_checks():
    from backend.app.services.risk_service import RiskService

    svc = RiskService.__new__(RiskService)
    svc.evaluation_mode = "api"
    svc.llm = _StubLLM()

    out = svc.analyze_message("Tell me about Poland in 2027.")

    # Stage 1
    assert "prompt_risk" in out
    pr = out["prompt_risk"]
    assert 0.0 <= pr["prompt_risk"] <= 1.0
    assert pr["s2_time_risk"] > 0.0  # 2027 is post-cutoff

    # Stage 3
    assert "semantic_checks" in out
    sc = out["semantic_checks"]
    assert sc["semantic_novelty"] is None  # disabled in this test
    assert 0.0 <= sc["entity_expansion"] <= 1.0
    assert 0.0 <= sc["claim_density"] <= 1.0

    # Core HRI still present
    assert "hri" in out and 0.0 <= out["hri"] <= 1.0
