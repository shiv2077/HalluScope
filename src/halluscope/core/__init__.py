"""
Core module for HalluScope.

This module contains core functionality for LLM interaction,
prompt management, and text processing utilities.
"""

from .llm_backend import LLMBackend
from .prompt_skeleton import PromptTemplate
from .text_utils import clean_text, tokenize

__all__ = ["LLMBackend", "PromptTemplate", "clean_text", "tokenize"]
