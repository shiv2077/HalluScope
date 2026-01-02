"""
Interventions Module.

This module provides intervention triggers and messages based on
risk levels to help users make informed decisions.
"""

from typing import Dict, Optional
from .thresholds import RiskLevel


class InterventionManager:
    """
    Manager for risk-based interventions.
    """
    
    DEFAULT_MESSAGES = {
        RiskLevel.LOW: "✅ Response confidence is high. The answer appears reliable.",
        RiskLevel.MEDIUM: "⚠️ Moderate uncertainty detected. Please verify critical facts before relying on this response.",
        RiskLevel.HIGH: "🔶 High hallucination risk! This response may contain inaccuracies. Cross-check with reliable sources.",
        RiskLevel.CRITICAL: "🛑 Critical risk detected! This response is highly likely to contain hallucinations. Do not rely on this information.",
    }
    
    def __init__(
        self,
        warning_threshold: float = 0.5,
        confirmation_threshold: float = 0.7,
        block_threshold: float = 0.9,
        custom_messages: Optional[Dict[RiskLevel, str]] = None
    ):
        """
        Initialize intervention manager.
        
        Args:
            warning_threshold: HRI threshold for showing warnings.
            confirmation_threshold: HRI threshold for requiring confirmation.
            block_threshold: HRI threshold for blocking response.
            custom_messages: Optional custom intervention messages.
        """
        self.warning_threshold = warning_threshold
        self.confirmation_threshold = confirmation_threshold
        self.block_threshold = block_threshold
        self.messages = custom_messages or self.DEFAULT_MESSAGES
    
    def get_intervention(self, hri: float, level: RiskLevel) -> Dict:
        """
        Get intervention details for given risk.
        
        Args:
            hri: HRI score.
            level: Risk level.
            
        Returns:
            Dictionary with intervention details.
        """
        # TODO: Implement intervention logic
        intervention = {
            "show_warning": hri >= self.warning_threshold,
            "require_confirmation": hri >= self.confirmation_threshold,
            "block_response": hri >= self.block_threshold,
            "message": self.messages.get(level, ""),
            "level": level.value,
        }
        return intervention
    
    def should_warn(self, hri: float) -> bool:
        """Check if warning should be shown."""
        return hri >= self.warning_threshold
    
    def should_confirm(self, hri: float) -> bool:
        """Check if confirmation is required."""
        return hri >= self.confirmation_threshold
    
    def should_block(self, hri: float) -> bool:
        """Check if response should be blocked."""
        return hri >= self.block_threshold


def get_intervention(hri: float, level: RiskLevel) -> Dict:
    """
    Convenience function to get intervention.
    
    Args:
        hri: HRI score.
        level: Risk level.
        
    Returns:
        Intervention dictionary.
    """
    manager = InterventionManager()
    return manager.get_intervention(hri, level)


def format_risk_display(hri: float, level: RiskLevel) -> str:
    """
    Format risk information for display.
    
    Args:
        hri: HRI score.
        level: Risk level.
        
    Returns:
        Formatted display string.
    """
    # TODO: Implement display formatting
    return f"Risk Level: {level.value.upper()} (HRI: {hri:.2f})"
