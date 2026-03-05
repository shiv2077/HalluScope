"""
HalluScope V5 — Stage 1: Pre-Hoc Prompt Risk Analysis

Estimates whether a prompt is likely to induce hallucination
BEFORE any tokens are generated.  100% database-free.

Signals:
1. Citation Pattern Score — detects academic/paper reference structures
2. Entity Density — counts named entities and specific references
3. Temporal Risk — detects dates beyond model training cutoff
4. Specificity Score — ratio of specific details to general tokens
5. Prompt Perplexity — how surprised the model is by the prompt (from prefill)
6. Entity Rarity — how unknown the rarest tokens are to the model

Output: prompt_risk_score ∈ [0, 1]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Model training cutoff — dates beyond this are suspicious
_TRAINING_CUTOFF_YEAR = 2024

# Citation-style patterns (regex, case-insensitive)
_CITATION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"summarize\s+(?:the\s+)?(?:key\s+)?(?:findings?\s+)?(?:of\s+)?(?:the\s+)?(?:paper|study|article|research)",
        r"the\s+\d{4}\s+(?:paper|study|article|book|publication)",
        r"published\s+in\s+(?:nature|science|IEEE|ACM|proceedings|journal|arxiv)",
        r"(?:by|from)\s+(?:dr\.|prof\.|professor|researcher)\s+",
        r"journal\s+of\s+\w+",
        r"(?:nature|science|IEEE|ACM)\s+\w+(?:\s+\w+)?(?:,\s*\d{4})?",
        r"explain\s+(?:the\s+)?(?:\w+\s+){1,4}?(?:theorem|conjecture|hypothesis|principle|law)\b",
        r"describe\s+the\s+(?:\w+\s+){1,4}?(?:experiment|method|technique|algorithm)\b",
        r"key\s+findings?\s+of",
        r"what\s+(?:did|does|were)\s+(?:the\s+)?(?:paper|study|research)\s+",
        r"according\s+to\s+(?:dr\.|prof\.|professor)\s+",
        r"(?:cite|reference|discuss)\s+(?:the\s+)?(?:work|paper|study)\s+(?:of|by)",
    ]
]

# Patterns for quoted titles (strong citation signal)
_QUOTED_TITLE_PATTERN = re.compile(
    r'["\u201c\u2018]([^"\u201d\u2019]{10,})["\u201d\u2019]'
)

# Temporal pattern — years
_YEAR_PATTERN = re.compile(r'\b(1[0-9]{3}|2[0-9]{3})\b')

# Named entity heuristic — capitalized multi-word sequences
_NAMED_ENTITY_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
)

# Weights for combining pre-hoc signals
PREHOC_WEIGHTS: Dict[str, float] = {
    "w_citation": 0.20,
    "w_entity_density": 0.08,
    "w_temporal": 0.12,
    "w_specificity": 0.05,
    "w_perplexity": 0.25,       # slightly reduced — short prompts bias high
    "w_rarity": 0.12,           # reduced — single rare token shouldn't dominate
    "w_quoted_title": 0.10,
    "bias": -2.5,               # more conservative baseline
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreHocResult:
    """Result of Stage 1 pre-hoc prompt risk analysis."""
    prompt_risk_score: float = 0.0

    # Individual signals
    citation_score: float = 0.0
    entity_density: float = 0.0
    temporal_risk: float = 0.0
    specificity_score: float = 0.0
    prompt_perplexity: float = 1.0
    entity_rarity: float = 0.0       # perplexity of rarest entity token
    quoted_title_score: float = 0.0
    low_prob_fraction: float = 0.0    # fraction of tokens with very low logprob

    # Diagnostic
    citation_patterns_matched: List[str] = field(default_factory=list)
    entities_found: List[str] = field(default_factory=list)
    years_found: List[int] = field(default_factory=list)
    prompt_risk_bias: float = 0.0     # bias to inject into Stage 2


# ─────────────────────────────────────────────────────────────────────────────
# Pure Pattern Analysis (no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def detect_citation_patterns(prompt: str) -> tuple[float, list[str]]:
    """Detect academic citation-style patterns in the prompt.

    Returns (score ∈ [0,1], list of matched pattern descriptions).
    """
    matches = []
    for pattern in _CITATION_PATTERNS:
        if pattern.search(prompt):
            matches.append(pattern.pattern[:60])

    # Quoted title is a very strong signal
    quoted = _QUOTED_TITLE_PATTERN.findall(prompt)
    if quoted:
        matches.append(f"quoted_title: {quoted[0][:40]}")

    # Normalize: 1 match = moderate, 2+ = high
    if not matches:
        return 0.0, []
    return min(1.0, len(matches) * 0.35), matches


def detect_quoted_titles(prompt: str) -> float:
    """Detect quoted titles — a strong fabrication prompt signal."""
    quoted = _QUOTED_TITLE_PATTERN.findall(prompt)
    if not quoted:
        return 0.0
    # Longer titles are more specific → higher risk
    max_len = max(len(t) for t in quoted)
    return min(1.0, max_len / 40.0)


def compute_entity_density(prompt: str) -> tuple[float, list[str]]:
    """Count named entities using capitalization heuristics.

    Returns (density ∈ [0,1], list of entities found).
    """
    entities = _NAMED_ENTITY_PATTERN.findall(prompt)
    # Also count standalone capitalized words (not sentence starters)
    words = prompt.split()
    standalone_caps = []
    for i, w in enumerate(words):
        if i > 0 and w[0].isupper() and w.isalpha() and len(w) > 2:
            standalone_caps.append(w)

    all_entities = list(set(entities + standalone_caps))
    if not words:
        return 0.0, []
    density = min(1.0, len(all_entities) / max(len(words) * 0.3, 1.0))
    return density, all_entities


def detect_temporal_risk(prompt: str) -> tuple[float, list[int]]:
    """Detect references to dates beyond model training cutoff.

    Returns (risk ∈ [0,1], list of years found).
    """
    years = [int(y) for y in _YEAR_PATTERN.findall(prompt)]
    if not years:
        return 0.0, []

    max_year = max(years)
    if max_year > _TRAINING_CUTOFF_YEAR:
        # Strong risk — model cannot know about this
        return min(1.0, (max_year - _TRAINING_CUTOFF_YEAR) * 0.3), years
    elif max_year < 1800:
        # Very old dates — obscure historical claims
        return 0.2, years
    return 0.0, years


def compute_specificity_score(prompt: str) -> float:
    """Measure how specific/detailed the prompt is.

    Very specific prompts (names, dates, titles, numbers) are more
    likely to trigger hallucination than general questions.
    """
    # Common question starters — NOT specific
    _QUESTION_WORDS = frozenset({
        "what", "who", "when", "where", "why", "how", "which",
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "should", "will", "shall",
        "explain", "describe", "tell", "list", "define",
        "the", "a", "an", "of", "in", "to", "for", "and",
    })

    words = prompt.split()
    if not words:
        return 0.0

    specific_count = 0
    for w in words:
        w_lower = w.strip(".,!?;:()\"'").lower()
        if w_lower in _QUESTION_WORDS:
            continue
        # Numbers (but not single digits in common context)
        if re.match(r'\d{2,}', w):
            specific_count += 1
        # Capitalized and not at sentence start (entity-like)
        elif w[0].isupper() and len(w) > 2 and w.isalpha() and w_lower not in _QUESTION_WORDS:
            specific_count += 1
        # Hyphenated compounds (e.g., "Quantum-Enhanced")
        elif '-' in w and len(w) > 5:
            specific_count += 1

    return min(1.0, specific_count / max(len(words) * 0.5, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Pattern-Only Analysis (runs BEFORE generation)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prompt_patterns(prompt: str) -> PreHocResult:
    """Run all pattern-based pre-hoc signals on the prompt.

    This can be called before generation — no model needed.
    The prompt_perplexity and entity_rarity fields will be filled
    later from the prefill logits.
    """
    citation_score, citation_matches = detect_citation_patterns(prompt)
    quoted_score = detect_quoted_titles(prompt)
    entity_density, entities = compute_entity_density(prompt)
    temporal_risk, years = detect_temporal_risk(prompt)
    specificity = compute_specificity_score(prompt)

    return PreHocResult(
        citation_score=citation_score,
        entity_density=entity_density,
        temporal_risk=temporal_risk,
        specificity_score=specificity,
        quoted_title_score=quoted_score,
        citation_patterns_matched=citation_matches,
        entities_found=entities,
        years_found=years,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Pre-Hoc Risk (combines patterns + model perplexity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_prompt_risk(
    pattern_result: PreHocResult,
    prompt_perplexity: float,
    min_token_logprob: float,
    low_prob_fraction: float,
    weights: Optional[Dict[str, float]] = None,
) -> PreHocResult:
    """Compute the final pre-hoc prompt risk score.

    Combines pattern-based signals with model-based signals
    (prompt perplexity extracted from the prefill step).

    Args:
        pattern_result: Result from analyze_prompt_patterns()
        prompt_perplexity: exp(-mean(prompt_logprobs)) from prefill
        min_token_logprob: Most surprising token in the prompt
        low_prob_fraction: Fraction of tokens with logprob < -5.0
        weights: Override default pre-hoc weights

    Returns:
        Updated PreHocResult with final prompt_risk_score
    """
    w = weights or PREHOC_WEIGHTS

    # Normalize prompt perplexity to [0, 1]
    # Low perplexity (~1-10) = familiar content = low risk
    # High perplexity (>100) = very unfamiliar = high risk
    norm_ppl = min(1.0, math.log1p(prompt_perplexity) / 8.0)  # log(e^8) ≈ 2981

    # Entity rarity: how surprising is the most unknown token
    # min_logprob of -3 is normal, -10 is very rare
    entity_rarity = min(1.0, max(0.0, (-min_token_logprob - 3.0) / 7.0))

    # Compute raw score
    x = (
        w["w_citation"] * pattern_result.citation_score * 10
        + w["w_entity_density"] * pattern_result.entity_density * 10
        + w["w_temporal"] * pattern_result.temporal_risk * 10
        + w["w_specificity"] * pattern_result.specificity_score * 10
        + w["w_perplexity"] * norm_ppl * 10
        + w["w_rarity"] * entity_rarity * 10
        + w["w_quoted_title"] * pattern_result.quoted_title_score * 10
        + w["bias"]
    )

    prompt_risk = _sigmoid(x)

    # Update result
    result = PreHocResult(
        prompt_risk_score=prompt_risk,
        citation_score=pattern_result.citation_score,
        entity_density=pattern_result.entity_density,
        temporal_risk=pattern_result.temporal_risk,
        specificity_score=pattern_result.specificity_score,
        quoted_title_score=pattern_result.quoted_title_score,
        prompt_perplexity=prompt_perplexity,
        entity_rarity=entity_rarity,
        low_prob_fraction=low_prob_fraction,
        citation_patterns_matched=pattern_result.citation_patterns_matched,
        entities_found=pattern_result.entities_found,
        years_found=pattern_result.years_found,
        # Bias to inject into generation HRI: scale prompt_risk into a sigmoid bias
        prompt_risk_bias=prompt_risk * 2.0,
    )

    return result
