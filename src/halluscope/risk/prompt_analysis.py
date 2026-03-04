"""
Prompt Risk Analysis for HalluScope.

STAGE 1: PRE-HOC HALLUCINATION DETECTION
=========================================

Analyzes prompts BEFORE generation to estimate hallucination risk.
Detects patterns that commonly trigger confident hallucinations:
- Citation requests (papers, theorems, experiments)
- Specific entity queries (names, dates, places)
- Factual claims requiring verification
- Entity rarity (unknown/obscure named entities)
- Knowledge cutoff mismatch (future dates)
- Compositional novelty (unusual concept combinations)

This addresses the limitation where LLMs confidently hallucinate
about fabricated papers, authors, or scientific claims with LOW entropy.

HALLUCINATION TYPES DETECTED:
- Type 1: Invented Entity (model creates new entities)
- Type 2: Affirmed False Prompt Fact (prompt contains fake entity)
- Type 3: Fabricated Facts About Real Entity (real entity, fake claims)
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# High-Risk Prompt Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Citation/academic patterns - highest risk for confident hallucination
CITATION_PATTERNS = [
    r"\b(paper|article|publication|journal|thesis|dissertation)\b",
    r"\b(theorem|lemma|corollary|proof|conjecture)\b",
    r"\b(experiment|study|research|investigation|analysis)\b",
    r"\b(algorithm|method|technique|approach|framework)\b",
    r"\b(discovered|invented|proposed|introduced|developed)\s+by\b",
    r"\b(published|presented|submitted|appeared)\s+(in|at)\b",
    r"\bby\s+[A-Z][a-z]+\s+(and|et al|in)\b",  # Author patterns
    # Enhanced citation detection
    r'"[^"]+" by [A-Z][a-z]+',  # "Title" by Author
    r'paper by\b',
    r'study by\b',
    r'theorem by\b',
    r'research by\b',
]

# Year + entity patterns - often fabricated
YEAR_ENTITY_PATTERNS = [
    r"\b(18|19|20)\d{2}\b.*\b(paper|theorem|discovery|experiment)\b",
    r"\b(paper|theorem|discovery)\b.*\b(18|19|20)\d{2}\b",
    r"\"[^\"]+\"\s+by\s+[A-Z]",  # Quoted title + author
]

# Specificity patterns - very specific claims are risky
SPECIFICITY_PATTERNS = [
    r"\b(exactly|precisely|specifically)\b",
    r"\b(the\s+)?first\s+(person|scientist|researcher)\s+to\b",
    r"\b(equation|formula|constant|coefficient)\s+(is|was|equals)\b",
    r"\b(born|died|lived)\s+(in|on)\s+\d",  # Biographical dates
    # Detect specific citation requests (high hallucination risk)
    r'"[^"]{10,}"\s*(?:by|authored|written)',  # "Title" by Author
    r"'[^']{10,}'\s*(?:by|authored|written)",  # 'Title' by Author  
    r'by\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # by FirstName LastName
    r'\b(18|19|20)\d{2}\b[^.]*"[^"]+"',  # Year ... "Title"
    r'"[^"]+"[^.]*\b(18|19|20)\d{2}\b',  # "Title" ... Year
]

# High-risk unverifiable citation patterns (deserves its own category)
UNVERIFIABLE_CITATION_PATTERNS = [
    r'"[^"]{10,100}"\s*by\s*[A-Z][a-z]+\s+[A-Z][a-z]+',  # "Full Title" by First Last
    r'\b(18|19|20)\d{2}\b.*"[^"]+".*by\s+[A-Z]',  # Year + "Title" + by Author
    r'"[^"]+".*by\s+[A-Z][a-z]+.*\b(18|19|20)\d{2}\b',  # "Title" + by Author + Year  
    r'summarize.*"[^"]+"',  # summarize "specific title"
    r'explain.*"[^"]+".*by',  # explain "title" by
]

# Summary/explanation requests - often trigger confabulation
SUMMARY_PATTERNS = [
    r"\bsummarize\b",
    r"\bexplain\s+(the|this|how)\b",
    r"\bdescribe\s+(the|this|how)\b",
    r"\bwhat\s+(is|was|are|were)\s+the\b.*\b(paper|theorem|study)\b",
    r"\bwho\s+(discovered|invented|proposed|wrote)\b",
]

# Entity extraction patterns
ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Multi-word proper nouns
    r"\"([^\"]+)\"",  # Quoted titles
    r"'([^']+)'",  # Single-quoted titles
    r"\b[A-Z][a-z]+[-–][A-Z][a-z]+\b",  # Hyphenated names (Maxwell-Boltzmann)
]

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 NEW: Knowledge Cutoff & Entity Rarity Data
# ─────────────────────────────────────────────────────────────────────────────

# Model knowledge cutoff year (configurable per model)
MODEL_KNOWLEDGE_CUTOFF = 2024

# Well-known entities for rarity calculation
WELL_KNOWN_SCIENTISTS = {
    "einstein", "newton", "darwin", "curie", "hawking", "feynman", "turing",
    "bohr", "heisenberg", "schrodinger", "maxwell", "planck", "dirac", "lorentz",
    "tesla", "edison", "galileo", "copernicus", "kepler", "faraday", "gauss",
    "euler", "pythagoras", "archimedes", "fermat", "riemann", "hilbert", "godel",
    "von neumann", "shannon", "chomsky", "knuth", "dijkstra", "ritchie", "torvalds",
}

WELL_KNOWN_CONCEPTS = {
    "relativity", "evolution", "quantum", "gravity", "electromagnetism",
    "thermodynamics", "entropy", "algorithm", "computation", "neural network",
    "machine learning", "deep learning", "calculus", "probability", "statistics",
    "dna", "gene", "protein", "cell", "neuron", "brain", "consciousness",
}

# Semantically distant concept pairs (for compositional novelty)
DISTANT_CONCEPT_PAIRS = [
    ({"quantum", "physics", "particle"}, {"consciousness", "philosophy", "mind"}),
    ({"mathematics", "theorem", "proof"}, {"biology", "evolution", "species"}),
    ({"computer", "algorithm", "software"}, {"astrology", "horoscope", "zodiac"}),
    ({"chemistry", "molecular", "reaction"}, {"linguistics", "grammar", "syntax"}),
]


@dataclass
class PromptRiskAnalysis:
    """Result of Stage 1 pre-hoc prompt risk analysis."""
    
    # Overall prompt risk score [0, 0.6]
    risk_score: float
    
    # Component scores (original)
    citation_risk: float       # Risk from citation patterns [0, 0.15]
    specificity_risk: float    # Risk from overly specific claims [0, 0.15]
    summary_risk: float        # Risk from summary requests [0, 0.10]
    unverifiable_risk: float   # Risk from unverifiable citation requests [0, 0.20]
    
    # NEW Stage 1 signals
    entity_rarity: float       # Rarity of named entities [0, 0.15]
    cutoff_risk: float         # Knowledge cutoff mismatch [0, 0.20]
    compositional_novelty: float  # Unusual concept combinations [0, 0.15]
    
    # Detected patterns
    detected_patterns: List[str]
    
    # Extracted entities from prompt (for entity novelty tracking)
    prompt_entities: Set[str]
    
    # Risk category
    risk_category: str  # "low", "medium", "high"
    
    # Explanation
    explanation: str


def compute_prompt_risk(prompt: str) -> PromptRiskAnalysis:
    """
    Analyze prompt to estimate pre-generation hallucination risk.
    
    This is TRUE PRE-HOC detection - computed BEFORE any tokens are generated.
    
    Args:
        prompt: The user's input prompt.
        
    Returns:
        PromptRiskAnalysis with risk score and breakdown.
    """
    prompt_lower = prompt.lower()
    detected_patterns: List[str] = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Citation Risk (0 - 0.15)
    # ─────────────────────────────────────────────────────────────────────────
    
    citation_hits = 0
    for pattern in CITATION_PATTERNS:
        if re.search(pattern, prompt_lower):
            citation_hits += 1
            detected_patterns.append(f"citation:{pattern[:30]}")
    
    # Year + entity patterns are especially risky
    for pattern in YEAR_ENTITY_PATTERNS:
        if re.search(pattern, prompt_lower):
            citation_hits += 2  # Weight these higher
            detected_patterns.append(f"year_entity:{pattern[:30]}")
    
    citation_risk = min(0.15, citation_hits * 0.03)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Specificity Risk (0 - 0.15) - INCREASED from 0.10
    # ─────────────────────────────────────────────────────────────────────────
    
    specificity_hits = 0
    for pattern in SPECIFICITY_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):  # Case insensitive
            specificity_hits += 1
            detected_patterns.append(f"specific:{pattern[:30]}")
    
    specificity_risk = min(0.15, specificity_hits * 0.04)  # Increased weight
    
    # ─────────────────────────────────────────────────────────────────────────
    # Unverifiable Citation Risk (0 - 0.20) - NEW HIGH-RISK CATEGORY
    # ─────────────────────────────────────────────────────────────────────────
    
    unverifiable_hits = 0
    for pattern in UNVERIFIABLE_CITATION_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            unverifiable_hits += 1
            detected_patterns.append(f"unverifiable:{pattern[:35]}")
    
    # This is the most dangerous pattern - specific title + author w/o verification
    unverifiable_risk = min(0.20, unverifiable_hits * 0.08)  # Strong weight
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary Request Risk (0 - 0.10)
    # ─────────────────────────────────────────────────────────────────────────
    
    summary_hits = 0
    for pattern in SUMMARY_PATTERNS:
        if re.search(pattern, prompt_lower):
            summary_hits += 1
            detected_patterns.append(f"summary:{pattern[:30]}")
    
    summary_risk = min(0.10, summary_hits * 0.04)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Extract Entities from Prompt
    # ─────────────────────────────────────────────────────────────────────────
    
    prompt_entities: Set[str] = set()
    for pattern in ENTITY_PATTERNS:
        matches = re.findall(pattern, prompt)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            # Filter out common words
            if len(match) > 2 and match.lower() not in COMMON_WORDS:
                prompt_entities.add(match)
    
    # ─────────────────────────────────────────────────────────────────────────
    # NEW STAGE 1: Entity Rarity Score (0 - 0.15)
    # ─────────────────────────────────────────────────────────────────────────
    
    entity_rarity = _compute_entity_rarity(prompt, prompt_entities)
    if entity_rarity > 0.05:
        detected_patterns.append(f"entity_rarity:{entity_rarity:.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # NEW STAGE 1: Knowledge Cutoff Risk (0 - 0.20)
    # ─────────────────────────────────────────────────────────────────────────
    
    cutoff_risk = _compute_cutoff_risk(prompt)
    if cutoff_risk > 0:
        detected_patterns.append(f"cutoff_risk:{cutoff_risk:.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # NEW STAGE 1: Compositional Novelty (0 - 0.15)
    # ─────────────────────────────────────────────────────────────────────────
    
    compositional_novelty = _compute_compositional_novelty(prompt)
    if compositional_novelty > 0.05:
        detected_patterns.append(f"compositional_novelty:{compositional_novelty:.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute Total Risk
    # ─────────────────────────────────────────────────────────────────────────
    
    total_risk = (
        citation_risk + 
        specificity_risk + 
        summary_risk + 
        unverifiable_risk +
        entity_rarity +
        cutoff_risk +
        compositional_novelty
    )
    
    # Cap at 0.60 (increased to accommodate new signals)
    total_risk = min(0.60, total_risk)
    
    # Categorize risk
    if total_risk >= 0.20:
        risk_category = "high"
    elif total_risk >= 0.10:
        risk_category = "medium"
    else:
        risk_category = "low"
    
    # Generate explanation
    explanations = []
    if cutoff_risk > 0.10:
        explanations.append("🚨 Future date reference - model cannot have knowledge")
    if unverifiable_risk > 0.05:
        explanations.append("🚨 Unverifiable citation request - HIGH fabrication risk")
    if citation_risk > 0.05:
        explanations.append("⚠️ Citation/academic pattern detected - high fabrication risk")
    if entity_rarity > 0.08:
        explanations.append("⚠️ Unknown/rare entities - may be fabricated")
    if compositional_novelty > 0.08:
        explanations.append("⚠️ Unusual concept combination - high risk of confabulation")
    if specificity_risk > 0.03:
        explanations.append("⚠️ Highly specific query - may trigger confabulation")
    if summary_risk > 0.03:
        explanations.append("⚠️ Summary request - model may invent details")
    
    explanation = "; ".join(explanations) if explanations else "✅ No high-risk patterns detected"
    
    return PromptRiskAnalysis(
        risk_score=round(total_risk, 4),
        citation_risk=round(citation_risk, 4),
        specificity_risk=round(specificity_risk, 4),
        summary_risk=round(summary_risk, 4),
        unverifiable_risk=round(unverifiable_risk, 4),
        entity_rarity=round(entity_rarity, 4),
        cutoff_risk=round(cutoff_risk, 4),
        compositional_novelty=round(compositional_novelty, 4),
        detected_patterns=detected_patterns,
        prompt_entities=prompt_entities,
        risk_category=risk_category,
        explanation=explanation,
    )


# Common words to exclude from entity extraction
COMMON_WORDS = {
    "the", "a", "an", "in", "on", "at", "by", "for", "to", "of", "and", "or",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "can", "but", "with", "from", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "any",
    # Common technical words
    "paper", "theorem", "study", "research", "experiment", "method", "algorithm",
    "published", "discovered", "invented", "proposed", "developed", "introduced",
}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 HELPER FUNCTIONS: New Pre-Hoc Signals
# ─────────────────────────────────────────────────────────────────────────────

def _compute_entity_rarity(prompt: str, prompt_entities: Set[str]) -> float:
    """
    Compute entity rarity score - how obscure/unknown are the named entities.
    
    Entities that are not in our well-known lists are considered rare and
    potentially fabricated (Type 2 hallucination risk).
    
    Args:
        prompt: Original user prompt
        prompt_entities: Extracted entities from prompt
        
    Returns:
        Rarity score [0, 0.15]
    """
    if not prompt_entities:
        return 0.0
    
    prompt_lower = prompt.lower()
    
    # Count unknown entities
    unknown_count = 0
    total_entities = len(prompt_entities)
    
    for entity in prompt_entities:
        entity_lower = entity.lower()
        # Check if entity is in our known lists
        is_known = any(known in entity_lower or entity_lower in known 
                       for known in WELL_KNOWN_SCIENTISTS)
        is_known = is_known or any(known in entity_lower 
                                    for known in WELL_KNOWN_CONCEPTS)
        
        if not is_known:
            unknown_count += 1
    
    # Base rarity score
    if total_entities > 0:
        rarity_ratio = unknown_count / total_entities
    else:
        rarity_ratio = 0.0
    
    rarity_score = rarity_ratio * 0.10
    
    # Boost if prompt contains academic context with unknown entities
    academic_context = any(term in prompt_lower for term in 
                           ["paper", "published", "theorem", "discovered", "by"])
    if academic_context and unknown_count > 0:
        rarity_score += 0.03
    
    # Boost for person names in "by X" pattern that aren't well-known
    author_pattern = r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    authors = re.findall(author_pattern, prompt)
    for author in authors:
        author_lower = author.lower()
        if not any(known in author_lower for known in WELL_KNOWN_SCIENTISTS):
            rarity_score += 0.02
    
    return min(0.15, rarity_score)


def _compute_cutoff_risk(prompt: str) -> float:
    """
    Detect knowledge cutoff mismatch - references to future dates.
    
    If prompt references years beyond model's knowledge cutoff,
    the model cannot have reliable information.
    
    Args:
        prompt: Original user prompt
        
    Returns:
        Cutoff risk score [0, 0.20]
    """
    current_year = datetime.now().year
    
    # Extract years from prompt
    year_matches = re.findall(r'\b(20\d{2})\b', prompt)
    
    future_years = []
    cutoff_risk = 0.0
    
    for year_str in year_matches:
        year = int(year_str)
        
        # Check if year is in the future or beyond knowledge cutoff
        if year > MODEL_KNOWLEDGE_CUTOFF:
            future_years.append(year)
            # Higher risk for further future years
            years_beyond = year - MODEL_KNOWLEDGE_CUTOFF
            cutoff_risk += min(0.10, 0.05 * years_beyond)
    
    # Additional risk if prompt explicitly asks about recent/future publications
    prompt_lower = prompt.lower()
    future_context_patterns = [
        r'(20\d{2})\s+(paper|publication|study|research)',
        r'(latest|recent|new|upcoming)\s+(paper|publication|study)',
        r'(explain|summarize).*\b20\d{2}\b',
    ]
    
    for pattern in future_context_patterns:
        if re.search(pattern, prompt_lower):
            cutoff_risk += 0.05
    
    return min(0.20, cutoff_risk)


def _compute_compositional_novelty(prompt: str) -> float:
    """
    Detect unusual combinations of concepts (compositional novelty).
    
    When prompt combines semantically distant domains, the model
    is more likely to confabulate connections that don't exist.
    
    Example: "Gödel's incompleteness theorem applied to quantum fluid harmonics"
    
    Args:
        prompt: Original user prompt
        
    Returns:
        Compositional novelty score [0, 0.15]
    """
    prompt_lower = prompt.lower()
    novelty_score = 0.0
    
    # Check for semantically distant concept pairs
    for domain1_concepts, domain2_concepts in DISTANT_CONCEPT_PAIRS:
        has_domain1 = any(concept in prompt_lower for concept in domain1_concepts)
        has_domain2 = any(concept in prompt_lower for concept in domain2_concepts)
        
        if has_domain1 and has_domain2:
            novelty_score += 0.08
    
    # Check for unusual modifier patterns
    unusual_patterns = [
        r'quantum\s+(consciousness|mind|soul|spirit)',
        r'(theorem|proof|equation)\s+applied\s+to\s+(biology|psychology|economics)',
        r'(harmonics|frequency)\s+in\s+(space|fluid|consciousness)',
        r'Einstein.*(?:dark matter|dark energy|quantum)',
        r'unified\s+theory\s+of\s+\w+\s+and\s+\w+',
    ]
    
    for pattern in unusual_patterns:
        if re.search(pattern, prompt_lower):
            novelty_score += 0.05
    
    # Check for made-up sounding technical terms
    fake_technical_patterns = [
        r'\b[A-Z][a-z]+ian\s+[A-Z][a-z]+ics\b',  # "Vallonian Harmonics"
        r'\b(meta|pseudo|quasi)-?[a-z]+\s+(field|theory|principle)\b',
        r'\b[A-Z][a-z]+-[A-Z][a-z]+\s+(theory|principle|effect)\b',  # "Vallon-Smith Theory"
    ]
    
    for pattern in fake_technical_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            novelty_score += 0.04
    
    return min(0.15, novelty_score)


def is_citation_request(prompt: str) -> bool:
    """Quick check if prompt is requesting citation information."""
    prompt_lower = prompt.lower()
    
    # Direct citation indicators
    citation_indicators = [
        "paper", "article", "publication", "thesis",
        "theorem", "lemma", "corollary",
        "summarize the", "explain the",
        "by " # Author attribution
    ]
    
    # Check for quoted title pattern
    has_quoted_title = bool(re.search(r'["\'][^"\']+["\']', prompt))
    
    # Check for year + topic pattern
    has_year_topic = bool(re.search(r'\b(18|19|20)\d{2}\b.*\b(paper|theorem|experiment|discovery)\b', prompt_lower))
    
    indicator_count = sum(1 for ind in citation_indicators if ind in prompt_lower)
    
    return indicator_count >= 2 or has_quoted_title or has_year_topic


def extract_claimed_entities(prompt: str) -> List[Tuple[str, str]]:
    """
    Extract entities that the prompt claims to exist.
    
    Returns list of (entity, type) tuples.
    """
    entities = []
    
    # Extract quoted titles
    titles = re.findall(r'["\']([^"\']+)["\']', prompt)
    for title in titles:
        entities.append((title, "title"))
    
    # Extract person names (capitalized sequences before/after "by")
    author_matches = re.findall(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', prompt)
    for author in author_matches:
        entities.append((author, "person"))
    
    # Extract years associated with claims
    year_matches = re.findall(r'\b((?:18|19|20)\d{2})\b', prompt)
    for year in year_matches:
        entities.append((year, "year"))
    
    return entities
