"""
Prompt Skeleton Module.

This module provides prompt template management for consistent
prompt formatting across different use cases and models.
"""

from typing import Optional


class PromptTemplate:
    """
    Template for formatting prompts.
    
    Provides consistent prompt formatting for queries with
    optional context injection.
    """
    
    # Default template
    DEFAULT_TEMPLATE = """Answer the following question truthfully and concisely.

Question: {query}

Answer:"""
    
    CONTEXT_TEMPLATE = """Use the following context to answer the question truthfully.

Context: {context}

Question: {query}

Answer:"""
    
    def __init__(self, template: Optional[str] = None):
        """
        Initialize prompt template.
        
        Args:
            template: Custom template string with {query} placeholder.
        """
        self.template = template or self.DEFAULT_TEMPLATE
    
    def format(self, query: str, context: Optional[str] = None) -> str:
        """
        Format a prompt with the given query and optional context.
        
        Args:
            query: The user's question or query.
            context: Optional context to include.
            
        Returns:
            Formatted prompt string.
        """
        # TODO: Implement prompt formatting
        if context:
            return self.CONTEXT_TEMPLATE.format(query=query, context=context)
        return self.template.format(query=query)
    
    def extract_response(self, output: str) -> str:
        """
        Extract the response portion from model output.
        
        Args:
            output: Raw model output.
            
        Returns:
            Cleaned response text.
        """
        # TODO: Implement response extraction
        return output.strip()


def build_chat_prompt(
    messages: list,
    system_prompt: Optional[str] = None
) -> str:
    """
    Build a chat-style prompt from message history.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        system_prompt: Optional system prompt.
        
    Returns:
        Formatted chat prompt.
    """
    # TODO: Implement chat prompt building
    pass
