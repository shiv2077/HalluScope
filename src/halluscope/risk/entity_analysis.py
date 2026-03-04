"""
Entity-Based Hallucination Detection for HalluScope.

Detects hallucinated entities by analyzing:
1. Novel entities introduced by the model (not in prompt)
2. Fabricated names, dates, titles that appear with high specificity
3. Entity patterns typical of hallucinated citations

This helps catch confident hallucinations where the model fabricates
specific names, papers, or facts with low entropy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Entity Extraction Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Person name patterns
PERSON_NAME_PATTERN = r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\b'

# Title patterns (in quotes)
TITLE_PATTERN = r'["\']([^"\']{5,})["\']'

# Year patterns
YEAR_PATTERN = r'\b((?:18|19|20)\d{2})\b'

# Institution patterns
INSTITUTION_PATTERN = r'\b(?:University|Institute|College|Laboratory|Center)\s+(?:of\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'

# Technical term patterns (often fabricated)
TECHNICAL_PATTERN = r'\b([A-Z][a-z]+(?:[-–][A-Z][a-z]+)+)\s+(?:theorem|principle|equation|constant|effect|law|rule|paradox|conjecture)\b'

# Journal/publication patterns
JOURNAL_PATTERN = r'\b(?:Journal|Proceedings|Transactions|Review|Letters|Bulletin)\s+(?:of\s+)?(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'


@dataclass
class EntityAnalysis:
    """Result of entity-based hallucination analysis."""
    
    # Novelty score [0, 0.25] - how many new entities were introduced
    novelty_score: float
    
    # Entities found only in output (not in prompt)
    novel_entities: List[str]
    
    # Entities found in both prompt and output
    grounded_entities: List[str]
    
    # High-risk fabrication patterns detected
    fabrication_signals: List[str]
    
    # Statistics
    total_entities_in_output: int
    novel_entity_ratio: float
    
    # Risk category
    risk_category: str


@dataclass  
class TokenEntityFlags:
    """Flags for individual tokens indicating entity-related risk."""
    
    is_entity: bool
    is_novel: bool        # Entity not in prompt
    is_fabricated: bool   # Matches fabrication patterns
    entity_type: Optional[str]  # "person", "title", "year", "institution", etc.
    risk_boost: float     # Additional risk for this token [0, 0.3]


def extract_entities(text: str) -> Dict[str, Set[str]]:
    """
    Extract named entities from text, categorized by type.
    
    Returns dict with keys: "persons", "titles", "years", "institutions", "technical"
    """
    entities: Dict[str, Set[str]] = {
        "persons": set(),
        "titles": set(),
        "years": set(),
        "institutions": set(),
        "technical": set(),
    }
    
    # Extract persons
    persons = re.findall(PERSON_NAME_PATTERN, text)
    for person in persons:
        # Filter out common false positives
        if not _is_common_phrase(person):
            entities["persons"].add(person.strip())
    
    # Extract titles (quoted strings)
    titles = re.findall(TITLE_PATTERN, text)
    for title in titles:
        if len(title) > 10:  # Substantial titles only
            entities["titles"].add(title.strip())
    
    # Extract years
    years = re.findall(YEAR_PATTERN, text)
    entities["years"] = set(years)
    
    # Extract institutions
    institutions = re.findall(INSTITUTION_PATTERN, text)
    entities["institutions"] = set(i.strip() for i in institutions)
    
    # Extract technical terms (often fabricated)
    technical = re.findall(TECHNICAL_PATTERN, text)
    entities["technical"] = set(t.strip() for t in technical)
    
    return entities


def compute_entity_novelty(
    prompt: str,
    generated_text: str,
    prompt_entities: Optional[Set[str]] = None,
) -> EntityAnalysis:
    """
    Compute entity novelty score by comparing prompt and output entities.
    
    High novelty = model introduced many new specific entities = higher fabrication risk.
    
    Args:
        prompt: Original user prompt
        generated_text: Model's generated response
        prompt_entities: Pre-extracted entities from prompt (optional)
        
    Returns:
        EntityAnalysis with novelty score and breakdown
    """
    # Extract entities from prompt
    if prompt_entities is None:
        prompt_ents = extract_entities(prompt)
        prompt_entities = set()
        for ent_set in prompt_ents.values():
            prompt_entities.update(ent_set)
    
    # Extract entities from generated text
    output_ents = extract_entities(generated_text)
    output_entities: Set[str] = set()
    for ent_set in output_ents.values():
        output_entities.update(ent_set)
    
    # Find novel entities (in output but not prompt)
    novel_entities = output_entities - prompt_entities
    grounded_entities = output_entities & prompt_entities
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute Novelty Score
    # ─────────────────────────────────────────────────────────────────────────
    
    total_output_entities = len(output_entities)
    
    if total_output_entities == 0:
        novelty_ratio = 0.0
    else:
        novelty_ratio = len(novel_entities) / total_output_entities
    
    # Base novelty score
    novelty_score = novelty_ratio * 0.15
    
    # ─────────────────────────────────────────────────────────────────────────
    # Detect Fabrication Signals
    # ─────────────────────────────────────────────────────────────────────────
    
    fabrication_signals: List[str] = []
    
    # Check for fabricated citation patterns
    # Novel person + novel year + novel title = very suspicious
    novel_persons = output_ents["persons"] - (extract_entities(prompt)["persons"] if prompt else set())
    novel_titles = output_ents["titles"] - (extract_entities(prompt)["titles"] if prompt else set())
    novel_years = output_ents["years"] - (extract_entities(prompt)["years"] if prompt else set())
    novel_technical = output_ents["technical"] - (extract_entities(prompt)["technical"] if prompt else set())
    
    if len(novel_persons) > 0 and len(novel_titles) > 0:
        fabrication_signals.append(f"⚠️ Novel author + novel title: possible fabricated citation")
        novelty_score += 0.05
    
    if len(novel_persons) > 0 and len(novel_years) > 0:
        fabrication_signals.append(f"⚠️ Novel person + novel year: possible fabricated biography")
        novelty_score += 0.03
    
    if len(novel_technical) > 0:
        fabrication_signals.append(f"⚠️ Novel technical term: '{list(novel_technical)[0]}' may be fabricated")
        novelty_score += 0.04
    
    # Check for suspicious specificity (many specific novel details)
    if len(novel_entities) >= 5:
        fabrication_signals.append(f"⚠️ High entity novelty: {len(novel_entities)} new specific entities")
        novelty_score += 0.03
    
    # Cap at 0.25
    novelty_score = min(0.25, novelty_score)
    
    # Categorize risk
    if novelty_score >= 0.15:
        risk_category = "high"
    elif novelty_score >= 0.08:
        risk_category = "medium"
    else:
        risk_category = "low"
    
    return EntityAnalysis(
        novelty_score=round(novelty_score, 4),
        novel_entities=list(novel_entities),
        grounded_entities=list(grounded_entities),
        fabrication_signals=fabrication_signals,
        total_entities_in_output=total_output_entities,
        novel_entity_ratio=round(novelty_ratio, 4),
        risk_category=risk_category,
    )


def get_token_entity_flags(
    token: str,
    token_position: int,
    all_tokens: List[str],
    prompt_entities: Set[str],
    output_entities: Set[str],
) -> TokenEntityFlags:
    """
    Determine if a token is part of a potentially fabricated entity.
    
    Used for token-level risk highlighting.
    """
    token_clean = token.strip()
    
    # Check if token is part of an entity
    is_entity = False
    is_novel = False
    is_fabricated = False
    entity_type = None
    risk_boost = 0.0
    
    # Check if capitalized (potential proper noun)
    if token_clean and token_clean[0].isupper():
        is_entity = True
        entity_type = "possible_proper_noun"
        
        # Check surrounding context for entity patterns
        context_start = max(0, token_position - 3)
        context_end = min(len(all_tokens), token_position + 3)
        context = " ".join(all_tokens[context_start:context_end])
        
        # Check if in a fabrication-prone context
        if any(kw in context.lower() for kw in ["paper", "published", "theorem", "discovered", "proposed"]):
            risk_boost = 0.15
            is_fabricated = True
        
        # Check if this entity appears in prompt
        # (simplified - checks if token itself is in prompt entities)
        if not any(token_clean in pe for pe in prompt_entities):
            is_novel = True
            risk_boost += 0.05
    
    # Check for year tokens in suspicious context
    if re.match(r'^(18|19|20)\d{2}$', token_clean):
        is_entity = True
        entity_type = "year"
        
        # Years in academic context are suspicious
        context_start = max(0, token_position - 5)
        context_end = min(len(all_tokens), token_position + 2)
        context = " ".join(all_tokens[context_start:context_end])
        
        if any(kw in context.lower() for kw in ["paper", "published", "discovered", "wrote"]):
            risk_boost = 0.10
            is_fabricated = True
    
    return TokenEntityFlags(
        is_entity=is_entity,
        is_novel=is_novel,
        is_fabricated=is_fabricated,
        entity_type=entity_type,
        risk_boost=risk_boost,
    )


def _is_common_phrase(text: str) -> bool:
    """Check if text is a common phrase (not a proper name)."""
    common_phrases = {
        "In The", "The First", "For Example", "As A", "In This",
        "This Is", "That Is", "It Is", "There Is", "There Are",
        "On The", "At The", "To The", "From The", "With The",
        "New York", "Los Angeles", "San Francisco",  # Keep real places
    }
    return text in common_phrases


# ─────────────────────────────────────────────────────────────────────────────
# Real-time Entity Tracking (for online monitoring)
# ─────────────────────────────────────────────────────────────────────────────

class EntityTracker:
    """
    Tracks entities during generation for real-time novelty monitoring.
    """
    
    def __init__(self, prompt_entities: Set[str]):
        self.prompt_entities = prompt_entities
        self.generated_entities: Set[str] = set()
        self.entity_count = 0
        self.novel_count = 0
    
    def add_token(self, token: str, all_tokens: List[str], position: int) -> float:
        """
        Process a new token and return incremental novelty risk.
        
        Returns:
            Risk increment for this token [0, 0.1]
        """
        flags = get_token_entity_flags(
            token, position, all_tokens, 
            self.prompt_entities, self.generated_entities
        )
        
        if flags.is_entity:
            self.entity_count += 1
            self.generated_entities.add(token.strip())
            
            if flags.is_novel:
                self.novel_count += 1
            
            return flags.risk_boost
        
        return 0.0
    
    @property
    def current_novelty_score(self) -> float:
        """Get current entity novelty score."""
        if self.entity_count == 0:
            return 0.0
        
        ratio = self.novel_count / self.entity_count
        return min(0.25, ratio * 0.15)
