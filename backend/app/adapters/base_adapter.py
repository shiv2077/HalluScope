"""
HalluScope – Base LLM Adapter

Defines the abstract interface that every model adapter must implement.
New adapters (Claude, Gemini, HuggingFace, etc.) should subclass
BaseLLMAdapter and provide a concrete `generate()` method that returns
a StandardizedResult.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from backend.app.adapters.schemas import StandardizedResult


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM adapters.

    Every adapter wraps a specific LLM provider and normalises the
    response into a StandardizedResult so downstream modules (metrics,
    risk engine) stay provider-agnostic.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> StandardizedResult:
        """Send a single prompt to the LLM."""
        ...

    @abstractmethod
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> StandardizedResult:
        """Send a multi-turn conversation to the LLM.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            max_tokens:  Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            A StandardizedResult for the assistant's reply.
        """
        ...
