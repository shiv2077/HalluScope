"""
HalluScope – Analysis Service

Orchestrates the full analysis pipeline:
    1. Receive the user prompt
    2. Call the selected LLM adapter
    3. Compute token-level metrics
    4. Compute the Hallucination Risk Index (HRI)
    5. Return a structured response

This service is the single entry-point used by the API layer and
keeps the endpoint handler thin and testable.
"""

from backend.app.adapters.base_adapter import BaseLLMAdapter
from backend.app.risk.hri_engine import compute_hri


def analyze_prompt(
    adapter: BaseLLMAdapter,
    prompt: str,
    max_tokens: int = 80,
    temperature: float = 0.7,
) -> dict:
    """Run the full HalluScope analysis pipeline.

    Args:
        adapter:     An instance of any BaseLLMAdapter subclass.
        prompt:      The user's input text.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        A dict ready to be serialised as the API JSON response:
        {
            "answer":     str,
            "hri":        float,
            "risk_level": str,
            "metrics": {
                "mean_logprob":      float,
                "logprob_std":       float,
                "confidence_decay":  float,
            },
            "model":      str,
        }
    """

    # 1 → Call the LLM via the adapter
    result = adapter.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # 2 → Feed log-probs into the HRI engine
    hri_result = compute_hri(result.token_logprobs)

    # 3 → Assemble the response
    return {
        "answer": result.text,
        "hri": hri_result["hri"],
        "risk_level": hri_result["risk_level"],
        "metrics": hri_result["metrics"],
        "model": result.model_name,
    }
