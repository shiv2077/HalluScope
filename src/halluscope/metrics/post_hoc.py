"""
HalluScope V5 — Stage 3: Post-Generation Semantic Verification

Detects confident hallucinations AFTER the full response is generated.
Catches fabrications that token-level signals (Stage 2) miss.

100% database-free — uses only embeddings and text statistics.

Signals:
1. Semantic Novelty — embedding distance between prompt and full answer
2. Entity Expansion — novel content words introduced in answer
3. Claim Density — assertion-to-evidence ratio in generated text
4. Confidence Paradox — high confidence × high novelty (KEY signal)
5. Narrative Template Score — fabrication language pattern detection
6. Concept Explosion — rate of new concept introduction

Output: verification_risk ∈ [0, 1]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "been", "be", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "of", "in", "to", "for", "with", "on", "at",
    "from", "by", "about", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "up", "down", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "their", "his",
    "her", "and", "or", "but", "not", "no", "if", "when", "while", "which",
    "what", "who", "whom", "how", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "also", "so", "then", "there", "here",
})

# Assertion patterns — the model making factual claims
_ASSERTION_PATTERNS = [
    "the paper demonstrated", "the paper proposed", "the paper showed",
    "the paper introduced", "the paper presents", "the paper describes",
    "the paper discusses", "the paper examines", "the paper explores",
    "the paper addresses", "the paper investigates", "the paper analyzes",
    "the paper outlines", "the paper builds on", "the paper focuses on",
    "the study demonstrated", "the study proposed", "the study showed",
    "the study found", "the study examines", "the study explores",
    "the research demonstrated", "the research showed", "the research found",
    "this work introduced", "this work developed", "this work established",
    "the theory states", "the theory explains", "the theory predicts",
    "the theory proposes", "the theory suggests", "the theorem states",
    "proved that", "showed that", "found that", "demonstrated that",
    "states that", "argues that", "suggests that", "claims that",
    "is known for", "is famous for", "is credited with", "was the first",
    "pioneered the", "formulated the", "theorized that", "hypothesized that",
    "the concept of", "a novel approach", "a new framework", "a new method",
    "key findings include", "the main contribution", "significant result",
    "breakthrough in", "revolutionized", "groundbreaking",
]

# Evidence / hedging patterns — model expressing uncertainty or citing sources
_EVIDENCE_PATTERNS = [
    "according to", "published in", "cited by", "referenced in",
    "doi:", "isbn:", "arxiv:", "et al.", "peer-reviewed",
    "journal", "proceedings", "volume", "experiment", "data shows",
    "evidence suggests", "empirical", "measured", "observed",
    "i don't know", "i'm not sure", "cannot confirm", "it is unclear",
    "it is uncertain", "however", "although", "on the other hand",
    "note that", "disclaimer", "not verified",
    "i cannot", "i'm unable", "no information", "not aware",
]

# Narrative fabrication templates — common structures in fabricated academic text
_FABRICATION_TEMPLATES: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"the (?:paper|study|research) (?:demonstrated|proposed|introduced|showed) (?:a|the|that)",
        r"(?:key|main|primary|significant) (?:findings?|contributions?|results?) (?:include|are|were)",
        r"this (?:work|approach|method|framework) (?:introduces?|proposes?|develops?|establishes?)",
        r"the (?:authors?|researchers?) (?:demonstrated|showed|found|proved|argued) that",
        r"the (?:theory|theorem|principle|concept) (?:states?|explains?|predicts?|suggests?) that",
        r"(?:experimental|empirical) results (?:show|demonstrate|indicate|suggest)",
        r"the (?:proposed|novel|new) (?:method|approach|framework|algorithm|model) (?:achieves?|outperforms?)",
        r"\d+%\s+(?:improvement|accuracy|reduction|increase)",
    ]
]


# Post-hoc weight profiles
POSTHOC_WEIGHTS: Dict[str, float] = {
    "w_novelty": 0.20,
    "w_entity": 0.03,          # entity expansion is noisy — reduce
    "w_claim": 0.20,
    "w_paradox": 0.30,         # confidence × novelty — strongest signal
    "w_template": 0.15,
    "w_explosion": 0.05,       # concept explosion is noisy — reduce
    "bias": -2.2,              # more conservative baseline
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PostHocResult:
    """Result of Stage 3 post-generation semantic verification."""
    verification_risk: float = 0.0

    # Individual signals
    semantic_novelty: float = 0.0
    entity_expansion: float = 0.0
    claim_density: float = 0.0
    confidence_paradox: float = 0.0
    narrative_template_score: float = 0.0
    concept_explosion: float = 0.0

    # Diagnostic
    assertions_found: int = 0
    evidence_found: int = 0
    templates_matched: List[str] = field(default_factory=list)
    new_concepts: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Signal Computation
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_semantic_novelty(
    prompt_embedding,
    answer_embedding,
) -> float:
    """Compute semantic novelty: 1 - cosine(prompt_emb, answer_emb).

    Uses pre-computed embeddings from the sentence transformer.
    High novelty = answer drifts far from prompt semantics.
    """
    import torch
    import torch.nn.functional as F

    if prompt_embedding is None or answer_embedding is None:
        return 0.0

    sim = F.cosine_similarity(
        prompt_embedding.unsqueeze(0).float(),
        answer_embedding.unsqueeze(0).float(),
        dim=1,
    ).item()
    sim = max(0.0, min(1.0, sim))
    return 1.0 - sim


def compute_entity_expansion(prompt: str, answer: str) -> tuple[float, list[str]]:
    """Fraction of answer content words NOT found in the prompt.

    Returns (expansion ∈ [0,1], list of novel concepts).
    Dampened for short prompts where high expansion is natural.
    """
    prompt_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', prompt))
    prompt_words -= _STOP_WORDS
    gen_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', answer))
    gen_content = gen_words - _STOP_WORDS
    if not gen_content:
        return 0.0, []
    new_content = gen_content - prompt_words
    raw_expansion = len(new_content) / max(len(gen_content), 1)

    # Dampen for short prompts: if prompt has < 5 content words,
    # high entity expansion is EXPECTED (answer naturally adds vocabulary).
    # Scale: 0 prompt words → 0.15x, 5+ → 1.0x
    n_prompt = len(prompt_words)
    dampen = min(1.0, n_prompt / 5.0) if n_prompt < 5 else 1.0
    expansion = min(1.0, raw_expansion * dampen)

    return expansion, sorted(new_content)[:20]


def compute_claim_density(answer: str) -> tuple[float, int, int]:
    """Ratio of assertive language to evidence/hedging markers.

    Returns (density ∈ [0,1], assertion_count, evidence_count).
    """
    text_lower = answer.lower()
    assertions = sum(1 for p in _ASSERTION_PATTERNS if p in text_lower)
    evidence = sum(1 for p in _EVIDENCE_PATTERNS if p in text_lower)
    if assertions + evidence == 0:
        return 0.0, 0, 0
    density = min(1.0, assertions / (assertions + evidence + 1))
    return density, assertions, evidence


def compute_confidence_paradox(confidence: float, novelty: float) -> float:
    """Detect the confidence paradox: high confidence + high novelty.

    This is the strongest signal for confident hallucinations.
    When the model is very confident (low entropy, high logprob)
    but producing semantically novel content (high embedding distance),
    it's almost certainly fabricating.

    Formula: paradox = confidence × novelty
    Range: [0, 1]
    """
    return max(0.0, min(1.0, confidence * novelty))


def compute_narrative_template_score(answer: str) -> tuple[float, list[str]]:
    """Detect fabrication narrative templates in the answer.

    Academic hallucinations follow characteristic patterns:
    "The paper demonstrated...", "Key findings include...", etc.
    """
    matches = []
    for pattern in _FABRICATION_TEMPLATES:
        found = pattern.findall(answer)
        if found:
            matches.append(pattern.pattern[:50])

    if not matches:
        return 0.0, []
    # 1 match = moderate, 3+ = high
    return min(1.0, len(matches) * 0.25), matches


def compute_concept_explosion(prompt: str, answer: str) -> float:
    """Measure how many new semantic clusters appear in the answer.

    concept_explosion = unique_concepts(answer) / unique_concepts(prompt)

    Large values indicate the model is inventing concepts that weren't
    in the prompt — a hallucination signal.
    Dampened for short prompts where concept growth is natural.
    """
    prompt_concepts = set(w.lower() for w in re.findall(r'\b\w{4,}\b', prompt))
    prompt_concepts -= _STOP_WORDS
    answer_concepts = set(w.lower() for w in re.findall(r'\b\w{4,}\b', answer))
    answer_concepts -= _STOP_WORDS

    new_concepts = answer_concepts - prompt_concepts
    n_prompt = len(prompt_concepts)

    if n_prompt == 0:
        # Very short prompt — heavily dampen (any answer adds concepts)
        return min(1.0, len(new_concepts) / 15.0) if new_concepts else 0.0

    ratio = len(new_concepts) / max(n_prompt, 1)

    # Dampen for short prompts: < 4 concepts → scale down
    dampen = min(1.0, n_prompt / 4.0) if n_prompt < 4 else 1.0
    return min(1.0, (ratio / 3.0) * dampen)  # normalize: 3x expansion → max risk


# ─────────────────────────────────────────────────────────────────────────────
# Full Post-Hoc Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    prompt: str,
    answer: str,
    confidence: float,
    semantic_stability: float,
    prompt_embedding=None,
    answer_embedding=None,
    weights: Optional[Dict[str, float]] = None,
) -> PostHocResult:
    """Run full Stage 3 post-generation verification.

    Args:
        prompt: Original user prompt
        answer: Generated text (without prompt prefix)
        confidence: Model confidence from generation (sigmoid(-mean_logprob))
        semantic_stability: Cosine(prompt_emb, answer_emb) from generation
        prompt_embedding: Prompt embedding tensor (from sentence transformer)
        answer_embedding: Answer embedding tensor
        weights: Override default post-hoc weights

    Returns:
        PostHocResult with verification_risk and all signal breakdowns
    """
    w = weights or POSTHOC_WEIGHTS

    # Signal 1: Semantic Novelty
    novelty = compute_semantic_novelty(prompt_embedding, answer_embedding)

    # Signal 2: Entity Expansion
    entity_exp, new_concepts_list = compute_entity_expansion(prompt, answer)

    # Signal 3: Claim Density
    claim_den, n_assert, n_evidence = compute_claim_density(answer)

    # Signal 4: Confidence Paradox (THE key signal)
    paradox = compute_confidence_paradox(confidence, novelty)

    # Signal 5: Narrative Template Detection
    template_score, templates = compute_narrative_template_score(answer)

    # Signal 6: Concept Explosion
    explosion = compute_concept_explosion(prompt, answer)

    # Combine into verification risk
    x = (
        w["w_novelty"] * novelty * 10
        + w["w_entity"] * entity_exp * 10
        + w["w_claim"] * claim_den * 10
        + w["w_paradox"] * paradox * 10
        + w["w_template"] * template_score * 10
        + w["w_explosion"] * explosion * 10
        + w["bias"]
    )

    verification_risk = _sigmoid(x)

    return PostHocResult(
        verification_risk=verification_risk,
        semantic_novelty=novelty,
        entity_expansion=entity_exp,
        claim_density=claim_den,
        confidence_paradox=paradox,
        narrative_template_score=template_score,
        concept_explosion=explosion,
        assertions_found=n_assert,
        evidence_found=n_evidence,
        templates_matched=templates,
        new_concepts=new_concepts_list,
    )
