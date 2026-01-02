"""
Application module for HalluScope.

This module provides the FastAPI backend, Streamlit UI,
request/response schemas, and service layer.
"""

# Note: Imports are lazy to avoid circular dependencies
# from .api import app
# from .ui import main as run_ui
# from .schemas import QueryRequest, RiskResponse
# from .service import HalluScopeService

__all__ = [
    "app",
    "run_ui",
    "QueryRequest",
    "RiskResponse",
    "HalluScopeService",
]
