"""HalluScope – OpenAI Adapter

Implements the BaseLLMAdapter interface for the OpenAI API.
Supports both single-prompt and multi-turn chat generation.

Logprobs are REQUIRED — the adapter will raise an error if the
chosen model does not return them.
"""

from typing import List, Dict
from openai import OpenAI

from backend.app.adapters.base_adapter import BaseLLMAdapter
from backend.app.adapters.schemas import StandardizedResult
from backend.app.config import OPENAI_API_KEY, DEFAULT_MODEL


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for the OpenAI API (gpt-4o-mini).

    Always requests token log-probabilities.  If the model does not
    support logprobs the API will return an error — this is intentional
    so we never silently degrade the analysis.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Single-prompt (backward compat) ──────────────────────────────
    def generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> StandardizedResult:
        return self.generate_chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ── Multi-turn chat ──────────────────────────────────────────────
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> StandardizedResult:
        """Call OpenAI Chat Completions with logprobs enabled."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        # ── Extract log-probabilities & token strings ────────────────
        token_logprobs: list[float] = []
        token_strings: list[str] = []
        top_logprobs: list[list[float]] = []

        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                token_logprobs.append(token_info.logprob)
                token_strings.append(token_info.token)
                if token_info.top_logprobs:
                    top_logprobs.append(
                        [alt.logprob for alt in token_info.top_logprobs]
                    )

        if not token_logprobs:
            raise RuntimeError(
                f"Model '{self.model}' did not return logprobs. "
                "Choose a model that supports logprobs (e.g. gpt-4o-mini)."
            )

        return StandardizedResult(
            text=text,
            token_logprobs=token_logprobs,
            token_strings=token_strings,
            top_logprobs=top_logprobs if top_logprobs else None,
            model_name=self.model,
        )
