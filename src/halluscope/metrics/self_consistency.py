"""
Self-Consistency Check for HalluScope.

Detects confident hallucinations by generating multiple responses
and checking if they agree. Hallucinated facts often change across
generations while true facts remain consistent.

This implements a lightweight version of semantic uncertainty measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import math


@dataclass
class ConsistencyAnalysis:
    """Result of self-consistency analysis."""
    
    # Consistency score [0, 1] - lower = more inconsistent = higher risk
    consistency_score: float
    
    # Risk from inconsistency [0, 0.25]
    consistency_risk: float
    
    # Number of samples compared
    num_samples: int
    
    # Key differences found
    differences: List[str]
    
    # Agreement metrics
    entity_agreement: float  # Do samples mention same entities?
    claim_agreement: float   # Do samples make same claims?
    
    # Risk category
    risk_category: str


def compute_text_similarity_simple(text1: str, text2: str) -> float:
    """
    Compute simple text similarity using word overlap.
    
    This is a lightweight alternative to sentence-transformers
    for faster execution. Production systems should use embeddings.
    
    Returns:
        Similarity score [0, 1]
    """
    # Tokenize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Remove common stopwords
    stopwords = {"the", "a", "an", "in", "on", "at", "by", "for", "to", "of", "and", "or", "is", "was", "are", "were"}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 or not words2:
        return 1.0  # Can't compare, assume consistent
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 1.0


def extract_factual_claims(text: str) -> Set[str]:
    """
    Extract factual claims from text for comparison.
    
    Looks for patterns like:
    - "X was born in Y"
    - "X published Y"
    - "In YEAR, X did Y"
    """
    claims = set()
    text_lower = text.lower()
    
    # Year-based claims
    year_claims = re.findall(r'in\s+((?:18|19|20)\d{2})\s*,?\s+([^.]{10,50})', text_lower)
    for year, claim in year_claims:
        claims.add(f"year_{year}:{claim[:40]}")
    
    # X did Y patterns
    action_claims = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(published|discovered|invented|proposed|wrote|developed)\s+([^.]{5,40})', text)
    for subject, action, obj in action_claims:
        claims.add(f"{subject}_{action}_{obj[:20]}")
    
    # Born/died claims
    bio_claims = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+was\s+(born|died)\s+(?:in|on)\s+([^.]{5,30})', text)
    for subject, event, detail in bio_claims:
        claims.add(f"{subject}_{event}_{detail[:20]}")
    
    return claims


def compute_entity_agreement(texts: List[str]) -> float:
    """
    Compute agreement on named entities across multiple texts.
    
    Returns:
        Agreement score [0, 1] - how much entity overlap exists
    """
    if len(texts) < 2:
        return 1.0
    
    # Extract entities from each text
    all_entities: List[Set[str]] = []
    for text in texts:
        entities = set()
        # Extract capitalized phrases (potential entities)
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        entities.update(matches)
        # Extract years
        years = re.findall(r'\b((?:18|19|20)\d{2})\b', text)
        entities.update(years)
        all_entities.append(entities)
    
    # Compute pairwise agreement
    agreements = []
    for i in range(len(all_entities)):
        for j in range(i + 1, len(all_entities)):
            e1, e2 = all_entities[i], all_entities[j]
            if e1 and e2:
                intersection = len(e1 & e2)
                union = len(e1 | e2)
                agreements.append(intersection / union if union > 0 else 1.0)
    
    return sum(agreements) / len(agreements) if agreements else 1.0


def compute_claim_agreement(texts: List[str]) -> float:
    """
    Compute agreement on factual claims across multiple texts.
    
    Returns:
        Agreement score [0, 1]
    """
    if len(texts) < 2:
        return 1.0
    
    # Extract claims from each text
    all_claims: List[Set[str]] = []
    for text in texts:
        claims = extract_factual_claims(text)
        all_claims.append(claims)
    
    # If no claims extracted, can't measure agreement
    total_claims = sum(len(c) for c in all_claims)
    if total_claims == 0:
        return 1.0  # No factual claims to disagree on
    
    # Compute agreement
    agreements = []
    for i in range(len(all_claims)):
        for j in range(i + 1, len(all_claims)):
            c1, c2 = all_claims[i], all_claims[j]
            if c1 and c2:
                intersection = len(c1 & c2)
                union = len(c1 | c2)
                agreements.append(intersection / union if union > 0 else 0.0)
            elif not c1 and not c2:
                agreements.append(1.0)  # Both have no claims - agree
            else:
                agreements.append(0.5)  # One has claims, one doesn't
    
    return sum(agreements) / len(agreements) if agreements else 1.0


def compute_self_consistency(
    responses: List[str],
    use_embeddings: bool = False,
) -> ConsistencyAnalysis:
    """
    Compute self-consistency across multiple model responses.
    
    Hallucinations often change details across generations.
    True facts tend to be consistent.
    
    Args:
        responses: List of generated responses to the same prompt
        use_embeddings: If True and sentence-transformers available, use embeddings
        
    Returns:
        ConsistencyAnalysis with risk score and breakdown
    """
    if len(responses) < 2:
        return ConsistencyAnalysis(
            consistency_score=1.0,
            consistency_risk=0.0,
            num_samples=len(responses),
            differences=[],
            entity_agreement=1.0,
            claim_agreement=1.0,
            risk_category="low",
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute Agreement Scores
    # ─────────────────────────────────────────────────────────────────────────
    
    # Try to use sentence-transformers if available and requested
    text_similarity = 1.0
    if use_embeddings:
        try:
            text_similarity = _compute_embedding_similarity(responses)
        except ImportError:
            # Fall back to simple similarity
            similarities = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarities.append(compute_text_similarity_simple(responses[i], responses[j]))
            text_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    else:
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarities.append(compute_text_similarity_simple(responses[i], responses[j]))
        text_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    
    # Entity agreement
    entity_agreement = compute_entity_agreement(responses)
    
    # Claim agreement
    claim_agreement = compute_claim_agreement(responses)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute Overall Consistency
    # ─────────────────────────────────────────────────────────────────────────
    
    # Weighted average of agreement metrics
    consistency_score = (
        0.3 * text_similarity +
        0.35 * entity_agreement +
        0.35 * claim_agreement
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Identify Differences
    # ─────────────────────────────────────────────────────────────────────────
    
    differences: List[str] = []
    
    if entity_agreement < 0.6:
        differences.append(f"⚠️ Entity disagreement: responses mention different names/entities")
    
    if claim_agreement < 0.5:
        differences.append(f"⚠️ Claim disagreement: responses make different factual claims")
    
    if text_similarity < 0.4:
        differences.append(f"⚠️ Low text overlap: responses are substantially different")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute Consistency Risk
    # ─────────────────────────────────────────────────────────────────────────
    
    # Lower consistency = higher risk
    # Map consistency [0, 1] to risk [0.25, 0]
    consistency_risk = (1.0 - consistency_score) * 0.25
    
    # Cap at 0.25
    consistency_risk = min(0.25, consistency_risk)
    
    # Categorize
    if consistency_risk >= 0.15:
        risk_category = "high"
    elif consistency_risk >= 0.08:
        risk_category = "medium"
    else:
        risk_category = "low"
    
    return ConsistencyAnalysis(
        consistency_score=round(consistency_score, 4),
        consistency_risk=round(consistency_risk, 4),
        num_samples=len(responses),
        differences=differences,
        entity_agreement=round(entity_agreement, 4),
        claim_agreement=round(claim_agreement, 4),
        risk_category=risk_category,
    )


def _compute_embedding_similarity(texts: List[str]) -> float:
    """
    Compute semantic similarity using sentence-transformers.
    
    Falls back to simple similarity if not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 1.0
        
    except ImportError:
        raise ImportError("sentence-transformers not available")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Consistency Check (for real-time use)
# ─────────────────────────────────────────────────────────────────────────────

def quick_consistency_check(response1: str, response2: str) -> Tuple[float, List[str]]:
    """
    Quick consistency check between two responses.
    
    Returns:
        Tuple of (consistency_risk, differences)
    """
    analysis = compute_self_consistency([response1, response2], use_embeddings=False)
    return analysis.consistency_risk, analysis.differences
