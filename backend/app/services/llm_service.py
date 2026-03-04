"""
HalluScope v2 – LLM Service

Wraps the adapter layer, maintains conversation history so the
LLM receives full chat context, and returns raw results for
downstream processing.
"""

from typing import List, Dict

from backend.app.adapters.openai_adapter import OpenAIAdapter
from backend.app.adapters.schemas import StandardizedResult
from backend.app.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


class LLMService:
    """Manages conversation state and delegates to the adapter."""

    def __init__(self):
        self.adapter = OpenAIAdapter()
        self.history: List[Dict[str, str]] = []

    def send_message(
        self,
        user_message: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> StandardizedResult:
        """Append the user message to history, call the LLM with
        full context, and return the StandardizedResult."""

        self.history.append({"role": "user", "content": user_message})

        result = self.adapter.generate_chat(
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

    def reset(self):
        self.history.clear()
