"""
In-Generation Monitoring for HalluScope.

STAGE 2: IN-GENERATION HALLUCINATION DETECTION
===============================================

Monitors the generation process in real-time to detect hallucination signals:
- Entity Commitment Risk: Detects when model commits to specific entities
- Semantic Drift: Detects when generation drifts from prompt semantics
- Pattern-based early warning signals

This module provides streaming-compatible analysis that runs during
token-by-token generation.

SIGNALS COMPUTED:
- entity_commitment_risk: Risk from committing to specific names/dates/citations
- semantic_drift: Distance between prompt and generated text embeddings
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Entity Commitment Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate model is "committing" to specific claims
COMMITMENT_PATTERNS = {
    "year": r'\b(18|19|20)\d{2}\b',
    "person_name": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
    "title": r'"[^"]{5,}"',  # Quoted title
    "citation": r'\([A-Z][a-z]+,?\s+(19|20)\d{2}\)',  # (Author, Year)
    "institution": r'University of [A-Z][a-z]+|[A-Z][a-z]+ Institute',
    "specific_number": r'\b\d+\.\d+\b',  # Specific decimals like 3.14159
    "date": r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b',
}

# Context patterns that increase commitment risk
HIGH_RISK_CONTEXT = [
    r'published\s+in\s+\d{4}',
    r'by\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+in\s+',
    r'discovered\s+by\s+',
    r'the\s+paper\s+"[^"]+"',
    r'(born|died)\s+on\s+',
    r'according\s+to\s+[A-Z][a-z]+',
]

# Patterns indicating hesitation (good sign - reduces risk)
HEDGING_PATTERNS = [
    r'\b(might|may|could|possibly|perhaps|potentially)\b',
    r'\b(uncertain|unclear|unknown|unverified)\b',
    r'\b(appears|seems|suggests)\b',
    r'\b(I think|I believe|I\'m not sure|I cannot verify)\b',
]


@dataclass
class EntityCommitmentResult:
    """Result of entity commitment analysis."""
    
    # Overall commitment risk [0, 0.25]
    commitment_risk: float
    
    # Detected commitments
    committed_entities: List[Tuple[str, str]]  # (entity, type)
    
    # Risk signals
    risk_signals: List[str]
    
    # Position-based risk (early commitment = higher risk)
    early_commitment_boost: float
    
    # Hedging detected (reduces risk)
    hedging_detected: bool


@dataclass
class SemanticDriftResult:
    """Result of semantic drift analysis."""
    
    # Drift score [0, 0.25]
    drift_score: float
    
    # Drift trajectoryg
    drift_trajectory: List[float]
    
    # Windowed drift (recent tokens)
    recent_drift: float
    
    # Topic shift detected
    topic_shift: bool


class GenerationMonitor:
    """
    Real-time monitor for in-generation hallucination detection.
    
    Tracks:
    - Entity commitment patterns over time
    - Semantic drift from original prompt
    - Sudden confidence spikes
    """
    
    def __init__(self, prompt: str, prompt_entities: Set[str] = None):
        """
        Initialize monitor with the original prompt.
        
        Args:
            prompt: The user's original prompt
            prompt_entities: Pre-extracted entities from prompt
        """
        self.prompt = prompt
        self.prompt_lower = prompt.lower()
        self.prompt_entities = prompt_entities or set()
        
        # Token tracking
        self.tokens: List[str] = []
        self.generated_text = ""
        
        # Entity commitment tracking
        self.committed_entities: List[Tuple[str, str, int]] = []  # (entity, type, position)
        self.commitment_risk = 0.0
        self.early_commitment_boost = 0.0
        
        # Semantic drift tracking
        self.drift_scores: List[float] = []
        self.topic_keywords: Set[str] = self._extract_topic_keywords(prompt)
        
        # Hedging detection
        self.hedging_count = 0
        
        # Risk signals
        self.risk_signals: List[str] = []
    
    def _extract_topic_keywords(self, text: str) -> Set[str]:
        """Extract topic-related keywords from text."""
        # Simple keyword extraction (for production, use TF-IDF or embeddings)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter out common words
        stopwords = {
            "the", "and", "that", "this", "with", "from", "have", "been",
            "what", "when", "where", "which", "about", "into", "more",
            "some", "would", "could", "should", "their", "there", "these",
        }
        return set(w for w in words if w not in stopwords)
    
    def add_token(self, token: str, entropy: float = 0.0, logprob: float = 0.0) -> Dict[str, float]:
        """
        Process a new token and return updated risk signals.
        
        Args:
            token: The newly generated token
            entropy: Token entropy (from model)
            logprob: Token log probability
            
        Returns:
            Dict with updated risk signals
        """
        self.tokens.append(token)
        self.generated_text = "".join(self.tokens)
        position = len(self.tokens)
        
        # ─────────────────────────────────────────────────────────────────────
        # Entity Commitment Detection
        # ─────────────────────────────────────────────────────────────────────
        
        commitment_increment = self._check_entity_commitment(position)
        
        # ─────────────────────────────────────────────────────────────────────
        # Hedging Detection (reduces risk)
        # ─────────────────────────────────────────────────────────────────────
        
        for pattern in HEDGING_PATTERNS:
            if re.search(pattern, self.generated_text[-50:].lower() if len(self.generated_text) > 50 else self.generated_text.lower()):
                self.hedging_count += 1
                break
        
        # ─────────────────────────────────────────────────────────────────────
        # Semantic Drift Detection
        # ─────────────────────────────────────────────────────────────────────
        
        drift_score = self._compute_drift()
        self.drift_scores.append(drift_score)
        
        # ─────────────────────────────────────────────────────────────────────
        # Compute Combined Signals
        # ─────────────────────────────────────────────────────────────────────
        
        # Apply hedging discount
        hedging_discount = min(0.10, self.hedging_count * 0.02)
        
        final_commitment_risk = max(0.0, self.commitment_risk - hedging_discount)
        
        return {
            "entity_commitment_risk": min(0.25, final_commitment_risk),
            "semantic_drift": min(0.25, drift_score),
            "hedging_detected": self.hedging_count > 0,
        }
    
    def _check_entity_commitment(self, position: int) -> float:
        """
        Check the latest text for entity commitment patterns.
        
        Returns risk increment for new commitments.
        """
        # Only check the recent portion of text
        recent_text = self.generated_text[-100:] if len(self.generated_text) > 100 else self.generated_text
        
        risk_increment = 0.0
        
        for entity_type, pattern in COMMITMENT_PATTERNS.items():
            matches = re.findall(pattern, recent_text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                # Check if this is a new commitment (not seen before)
                match_key = (match, entity_type)
                existing = [(e, t) for (e, t, p) in self.committed_entities]
                
                if match_key not in existing:
                    self.committed_entities.append((match, entity_type, position))
                    
                    # Compute risk based on position and type
                    base_risk = 0.02
                    
                    # Early commitments are riskier
                    if position < 20:
                        base_risk *= 1.5
                        self.early_commitment_boost += 0.01
                    
                    # Certain types are riskier
                    if entity_type in ["citation", "person_name", "date"]:
                        base_risk *= 1.3
                    
                    # Check if entity is in prompt (grounded)
                    is_grounded = any(match in pe for pe in self.prompt_entities)
                    if is_grounded:
                        base_risk *= 0.5  # Reduce risk for grounded entities
                    else:
                        # Novel entity - higher risk
                        base_risk *= 1.2
                        self.risk_signals.append(f"Novel {entity_type}: {match[:30]}")
                    
                    risk_increment += base_risk
        
        # Check for high-risk context patterns
        for pattern in HIGH_RISK_CONTEXT:
            if re.search(pattern, recent_text):
                risk_increment += 0.01
        
        self.commitment_risk += risk_increment
        self.commitment_risk = min(0.25, self.commitment_risk)
        
        return risk_increment
    
    def _compute_drift(self) -> float:
        """
        Compute semantic drift from prompt.
        
        Uses keyword overlap as a simple proxy for semantic similarity.
        For production, use embedding similarity.
        """
        if len(self.tokens) < 10:
            return 0.0
        
        # Extract keywords from recent generation
        window_size = min(30, len(self.tokens))
        recent_text = "".join(self.tokens[-window_size:]).lower()
        recent_keywords = self._extract_topic_keywords(recent_text)
        
        if not self.topic_keywords or not recent_keywords:
            return 0.0
        
        # Compute keyword overlap (inverse = drift)
        overlap = len(recent_keywords & self.topic_keywords)
        total = len(recent_keywords | self.topic_keywords)
        
        if total == 0:
            return 0.0
        
        similarity = overlap / total
        drift = 1.0 - similarity
        
        # Scale to [0, 0.25] range
        return min(0.25, drift * 0.3)
    
    def get_commitment_result(self) -> EntityCommitmentResult:
        """Get final entity commitment analysis."""
        # Apply hedging discount
        hedging_discount = min(0.10, self.hedging_count * 0.02)
        final_risk = max(0.0, self.commitment_risk - hedging_discount)
        
        return EntityCommitmentResult(
            commitment_risk=round(final_risk, 4),
            committed_entities=[(e, t) for (e, t, p) in self.committed_entities],
            risk_signals=self.risk_signals,
            early_commitment_boost=round(self.early_commitment_boost, 4),
            hedging_detected=self.hedging_count > 0,
        )
    
    def get_drift_result(self) -> SemanticDriftResult:
        """Get final semantic drift analysis."""
        if not self.drift_scores:
            return SemanticDriftResult(
                drift_score=0.0,
                drift_trajectory=[],
                recent_drift=0.0,
                topic_shift=False,
            )
        
        # Compute recent drift (last 20 tokens)
        recent_drift = sum(self.drift_scores[-20:]) / len(self.drift_scores[-20:]) if len(self.drift_scores) >= 20 else 0.0
        
        # Detect topic shift (sudden increase in drift)
        topic_shift = False
        if len(self.drift_scores) > 30:
            early_drift = sum(self.drift_scores[:15]) / 15
            late_drift = sum(self.drift_scores[-15:]) / 15
            if late_drift > early_drift + 0.1:
                topic_shift = True
        
        return SemanticDriftResult(
            drift_score=round(max(self.drift_scores) if self.drift_scores else 0.0, 4),
            drift_trajectory=self.drift_scores,
            recent_drift=round(recent_drift, 4),
            topic_shift=topic_shift,
        )


def compute_entity_commitment_risk(
    generated_text: str,
    prompt: str = "",
    prompt_entities: Set[str] = None,
    token_position: int = 0,
) -> EntityCommitmentResult:
    """
    Standalone function to compute entity commitment risk.
    
    Useful for batch analysis or post-generation checks.
    """
    monitor = GenerationMonitor(prompt, prompt_entities)
    
    # Process text as a single "token" for simplicity
    monitor.add_token(generated_text)
    
    return monitor.get_commitment_result()


def compute_semantic_drift(
    generated_text: str,
    prompt: str,
    window_size: int = 30,
) -> float:
    """
    Compute semantic drift between prompt and generation.
    
    Args:
        generated_text: The model's generated response
        prompt: The original user prompt
        window_size: Size of sliding window for drift computation
        
    Returns:
        Drift score [0, 0.25]
    """
    monitor = GenerationMonitor(prompt)
    
    # Split text into pseudo-tokens (words)
    words = generated_text.split()
    for word in words:
        monitor.add_token(word + " ")
    
    result = monitor.get_drift_result()
    return result.drift_score
