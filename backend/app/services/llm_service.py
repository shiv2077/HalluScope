"""
HalluScope v2 – LLM Service

Wraps the adapter layer, maintains conversation history so the
LLM receives full chat context, and returns raw results for
downstream processing.
"""

from typing import List, Dict

from backend.app.adapters.openai_adapter import OpenAIAdapter
from backend.app.adapters.base_adapter import BaseLLMAdapter
from backend.app.adapters.schemas import StandardizedResult
from backend.app.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


class LLMService:
    """Manages conversation state and delegates to the adapter."""

    def __init__(self, evaluation_mode: str = "api"):
        self.evaluation_mode = evaluation_mode
        self.adapter: BaseLLMAdapter | None = None
        self.history: List[Dict[str, str]] = []

    def _get_adapter(self) -> BaseLLMAdapter:
        if self.adapter is not None:
            return self.adapter

        if self.evaluation_mode == "local":
            try:
                from backend.app.adapters.local_adapter import LocalAdapter
            except Exception as exc:
                raise RuntimeError(
                    "Local mode dependencies are missing. Install backend requirements "
                    "to use local evaluation mode."
                ) from exc

            self.adapter = LocalAdapter()
        else:
            self.adapter = OpenAIAdapter()

        return self.adapter

    def send_message(
        self,
        user_message: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = DEFAULT_TEMPERATURE,
    ) -> StandardizedResult:
        """Append the user message to history, call the LLM with
        full context, and return the StandardizedResult.

        ``temperature=None`` falls back to the configured default; pass ``0``
        for greedy/deterministic decoding.
        """
        if temperature is None:
            temperature = DEFAULT_TEMPERATURE

        self.history.append({"role": "user", "content": user_message})

        result = self._get_adapter().generate_chat(
            messages=self.history,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Keep assistant reply in history for multi-turn context
        self.history.append({"role": "assistant", "content": result.text})
        return result

    def add_assistant_message(self, content: str):
        """Manually inject an assistant message (e.g. intervention)."""
        self.history.append({"role": "assistant", "content": content})

    def pop_last_assistant(self) -> None:
        """Remove the most recent assistant message from history if present.

        Used by the intervention flow: send_message() always appends the real
        LLM reply, but if the risk engine then triggers an intervention we must
        strip that reply before injecting the clarification — otherwise the
        suppressed answer remains in the conversation context for future turns.
        """
        if self.history and self.history[-1].get("role") == "assistant":
            self.history.pop()

    def reset(self):
        self.history.clear()
