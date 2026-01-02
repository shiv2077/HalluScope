"""
Threshold Classification Module.

This module provides threshold-based classification of HRI scores
into discrete risk levels for user-facing display.
"""

from enum import Enum
from typing import Dict, Optional


class RiskLevel(Enum):
    """Risk level categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThresholdClassifier:
    """
    Classifier for converting HRI scores to risk levels.
    """
    
    DEFAULT_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8,
    }
    
    COLORS = {
        RiskLevel.LOW: "#28a745",      # Green
        RiskLevel.MEDIUM: "#ffc107",   # Yellow
        RiskLevel.HIGH: "#fd7e14",     # Orange
        RiskLevel.CRITICAL: "#dc3545", # Red
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize threshold classifier.
        
        Args:
            thresholds: Custom threshold values.
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
    
    def classify(self, hri: float) -> RiskLevel:
        """
        Classify HRI score into risk level.
        
        Args:
            hri: HRI score between 0 and 1.
            
        Returns:
            RiskLevel enum value.
        """
        # TODO: Implement classification
        if hri < self.thresholds["low"]:
            return RiskLevel.LOW
        elif hri < self.thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif hri < self.thresholds["high"]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_color(self, level: RiskLevel) -> str:
        """
        Get display color for risk level.
        
        Args:
            level: RiskLevel enum value.
            
        Returns:
            Hex color code.
        """
        return self.COLORS.get(level, "#6c757d")
    
    def get_label(self, level: RiskLevel) -> str:
        """
        Get display label for risk level.
        
        Args:
            level: RiskLevel enum value.
            
        Returns:
            Human-readable label.
        """
        labels = {
            RiskLevel.LOW: "Low Risk",
            RiskLevel.MEDIUM: "Medium Risk",
            RiskLevel.HIGH: "High Risk",
            RiskLevel.CRITICAL: "Critical Risk",
        }
        return labels.get(level, "Unknown")


def classify_risk(hri: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    Convenience function to classify risk.
    
    Args:
        hri: HRI score.
        thresholds: Optional custom thresholds.
        
    Returns:
        Risk level string.
    """
    classifier = ThresholdClassifier(thresholds)
    return classifier.classify(hri).value
