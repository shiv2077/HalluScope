"""
Post-Hoc Verification for HalluScope.

STAGE 3: POST-GENERATION HALLUCINATION DETECTION
==================================================

Runs after generation completes to verify the response:
- Self-Consistency Check: Compare multiple generations
- Semantic Entropy: Cluster claims and compute entropy
- Citation Verification: Verify references using external APIs

This provides the final layer of hallucination detection.

SIGNALS COMPUTED:
- self_consistency_risk: Disagreement across multiple generations
- semantic_entropy: Entropy of claim clusters
- citation_verification_risk: Risk from unverifiable citations
"""

from __future__ import annotations

import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Claim Extraction Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Patterns to extract factual claims
CLAIM_PATTERNS = [
    r'([A-Z][^.!?]*(?:is|was|are|were|has|have|had)[^.!?]*\.)',  # Declarative statements
    r'([A-Z][^.!?]*(?:in|on|at)\s+\d{4}[^.!?]*\.)',  # Date claims
    r'([A-Z][^.!?]*(?:discovered|invented|proposed|published|found)[^.!?]*\.)',  # Discovery claims
    r'([A-Z][^.!?]*(?:according to|states that|shows that)[^.!?]*\.)',  # Attribution claims
]

# Citation patterns for verification
CITATION_PATTERNS = [
    (r'"([^"]+)"\s*(?:by|,)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', "title_author"),
    (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\((\d{4})\)', "author_year"),
    (r'(?:published|appeared)\s+in\s+([A-Z][^,]+)', "journal"),
]


@dataclass
class SelfConsistencyResult:
    """Result of self-consistency check."""
    
    # Consistency risk [0, 0.25] - higher = more inconsistent
    consistency_risk: float
    
    # Agreement score [0, 1]
    agreement_score: float
    
    # Number of generations compared
    num_generations: int
    
    # Key disagreements found
    disagreements: List[str]
    
    # Claim-level consistency
    claim_consistency: Dict[str, float]


@dataclass
class SemanticEntropyResult:
    """Result of semantic entropy computation."""
    
    # Semantic entropy [0, 0.25]
    semantic_entropy: float
    
    # Number of unique claims
    num_claims: int
    
    # Number of claim clusters
    num_clusters: int
    
    # Cluster distribution
    cluster_distribution: List[float]
    
    # High entropy claims
    uncertain_claims: List[str]


@dataclass
class CitationVerificationResult:
    """Result of citation verification."""
    
    # Verification risk [0, 0.25]
    verification_risk: float
    
    # Citations found
    citations_found: List[Dict[str, str]]
    
    # Verified citations
    verified_count: int
    
    # Unverified citations
    unverified_count: int
    
    # Verification details
    verification_details: List[str]


@dataclass
class PostHocAnalysis:
    """Combined post-hoc analysis results."""
    
    # Overall post-hoc risk [0, 0.50]
    posthoc_risk: float
    
    # Component results
    self_consistency: SelfConsistencyResult
    semantic_entropy: SemanticEntropyResult
    citation_verification: CitationVerificationResult
    
    # Summary
    risk_summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Known Citations Database (for lightweight verification)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_CITATIONS = {
    # Famous papers
    "origin of species": {"author": "darwin", "year": 1859},
    "principia": {"author": "newton", "year": 1687},
    "on the electrodynamics of moving bodies": {"author": "einstein", "year": 1905},
    "a brief history of time": {"author": "hawking", "year": 1988},
    "computing machinery and intelligence": {"author": "turing", "year": 1950},
    "attention is all you need": {"author": "vaswani", "year": 2017},
    "imagenet classification with deep convolutional neural networks": {"author": "krizhevsky", "year": 2012},
    "dropout": {"author": "srivastava", "year": 2014},
    "generative adversarial nets": {"author": "goodfellow", "year": 2014},
    "deep residual learning": {"author": "he", "year": 2016},
}


# ─────────────────────────────────────────────────────────────────────────────
# Self-Consistency Check
# ─────────────────────────────────────────────────────────────────────────────

def extract_claims(text: str) -> List[str]:
    """
    Extract factual claims from generated text.
    
    Returns list of claim strings.
    """
    claims = []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        # Check if sentence contains factual indicators
        factual_indicators = [
            r'\b(is|was|are|were|has|have|had)\b',
            r'\b(discovered|invented|proposed|published|found)\b',
            r'\b\d{4}\b',  # Years
            r'\b(according to|states that|shows that)\b',
        ]
        
        for pattern in factual_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                claims.append(sentence)
                break
    
    return claims


def compute_claim_similarity(claim1: str, claim2: str) -> float:
    """
    Compute similarity between two claims.
    
    Uses simple word overlap. For production, use sentence embeddings.
    """
    words1 = set(re.findall(r'\b\w{3,}\b', claim1.lower()))
    words2 = set(re.findall(r'\b\w{3,}\b', claim2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    total = len(words1 | words2)
    
    return overlap / total if total > 0 else 0.0


def check_self_consistency(
    generations: List[str],
    similarity_threshold: float = 0.7,
) -> SelfConsistencyResult:
    """
    Check self-consistency across multiple generations.
    
    Args:
        generations: List of generated responses for the same prompt
        similarity_threshold: Threshold for considering claims consistent
        
    Returns:
        SelfConsistencyResult with consistency analysis
    """
    if len(generations) < 2:
        return SelfConsistencyResult(
            consistency_risk=0.0,
            agreement_score=1.0,
            num_generations=len(generations),
            disagreements=[],
            claim_consistency={},
        )
    
    # Extract claims from each generation
    all_claims: List[List[str]] = []
    for gen in generations:
        claims = extract_claims(gen)
        all_claims.append(claims)
    
    # Compare claims across generations
    disagreements = []
    claim_consistency: Dict[str, float] = {}
    total_agreement = 0.0
    comparisons = 0
    
    # Compare first generation with others
    base_claims = all_claims[0]
    
    for claim in base_claims:
        consistencies = []
        
        for other_claims in all_claims[1:]:
            # Find most similar claim in other generation
            max_sim = 0.0
            for other_claim in other_claims:
                sim = compute_claim_similarity(claim, other_claim)
                max_sim = max(max_sim, sim)
            
            consistencies.append(max_sim)
            total_agreement += max_sim
            comparisons += 1
        
        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 1.0
        claim_consistency[claim[:50] + "..."] = avg_consistency
        
        if avg_consistency < similarity_threshold:
            disagreements.append(f"Inconsistent: '{claim[:60]}...'")
    
    # Compute overall agreement
    agreement_score = total_agreement / comparisons if comparisons > 0 else 1.0
    
    # Convert to risk score [0, 0.25]
    consistency_risk = (1.0 - agreement_score) * 0.25
    
    return SelfConsistencyResult(
        consistency_risk=round(consistency_risk, 4),
        agreement_score=round(agreement_score, 4),
        num_generations=len(generations),
        disagreements=disagreements[:5],  # Limit to top 5
        claim_consistency=claim_consistency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Entropy
# ─────────────────────────────────────────────────────────────────────────────

def cluster_claims(claims: List[str], threshold: float = 0.6) -> List[List[str]]:
    """
    Cluster similar claims together.
    
    Simple greedy clustering based on similarity.
    """
    if not claims:
        return []
    
    clusters: List[List[str]] = []
    assigned = set()
    
    for i, claim in enumerate(claims):
        if i in assigned:
            continue
        
        # Start new cluster
        cluster = [claim]
        assigned.add(i)
        
        # Find similar claims
        for j, other_claim in enumerate(claims):
            if j in assigned:
                continue
            
            sim = compute_claim_similarity(claim, other_claim)
            if sim >= threshold:
                cluster.append(other_claim)
                assigned.add(j)
        
        clusters.append(cluster)
    
    return clusters


def compute_semantic_entropy(
    text: str,
    generations: List[str] = None,
) -> SemanticEntropyResult:
    """
    Compute semantic entropy of claims.
    
    High entropy = many diverse, potentially contradictory claims.
    
    Args:
        text: Generated text to analyze
        generations: Optional list of multiple generations for cross-entropy
        
    Returns:
        SemanticEntropyResult with entropy analysis
    """
    # Extract all claims
    claims = extract_claims(text)
    
    if generations:
        for gen in generations:
            claims.extend(extract_claims(gen))
    
    if not claims:
        return SemanticEntropyResult(
            semantic_entropy=0.0,
            num_claims=0,
            num_clusters=0,
            cluster_distribution=[],
            uncertain_claims=[],
        )
    
    # Cluster claims
    clusters = cluster_claims(claims)
    
    # Compute cluster size distribution
    total = len(claims)
    cluster_probs = [len(c) / total for c in clusters]
    
    # Compute entropy: H = -sum(p * log(p))
    entropy = 0.0
    for p in cluster_probs:
        if p > 0:
            entropy -= p * math.log(p + 1e-12)
    
    # Normalize to [0, 1] based on maximum possible entropy
    max_entropy = math.log(len(claims)) if len(claims) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Scale to [0, 0.25]
    semantic_entropy_score = normalized_entropy * 0.25
    
    # Find uncertain claims (in small clusters)
    uncertain_claims = []
    for cluster in clusters:
        if len(cluster) == 1:
            uncertain_claims.append(cluster[0][:60] + "...")
    
    return SemanticEntropyResult(
        semantic_entropy=round(semantic_entropy_score, 4),
        num_claims=len(claims),
        num_clusters=len(clusters),
        cluster_distribution=[round(p, 3) for p in cluster_probs],
        uncertain_claims=uncertain_claims[:5],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Citation Verification
# ─────────────────────────────────────────────────────────────────────────────

def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Extract citations from text.
    
    Returns list of citation dicts with title, author, year, etc.
    """
    citations = []
    
    # Pattern: "Title" by/by Author
    title_author_pattern = r'"([^"]+)"\s*(?:by|,)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    for match in re.finditer(title_author_pattern, text):
        citations.append({
            "title": match.group(1),
            "author": match.group(2),
            "type": "title_author",
        })
    
    # Pattern: Author (Year)
    author_year_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\((\d{4})\)'
    for match in re.finditer(author_year_pattern, text):
        citations.append({
            "author": match.group(1),
            "year": match.group(2),
            "type": "author_year",
        })
    
    # Pattern: published in Journal
    journal_pattern = r'published\s+in\s+(?:the\s+)?([A-Z][^,\.]+)'
    for match in re.finditer(journal_pattern, text):
        citations.append({
            "journal": match.group(1).strip(),
            "type": "journal",
        })
    
    return citations


def verify_citation_local(citation: Dict[str, str]) -> Tuple[bool, str]:
    """
    Verify citation against local known citations database.
    
    Returns (verified, reason).
    """
    title = citation.get("title", "").lower()
    author = citation.get("author", "").lower().split()[-1] if citation.get("author") else ""
    year = citation.get("year", "")
    
    for known_title, info in KNOWN_CITATIONS.items():
        if known_title in title or title in known_title:
            # Title match - check author and year
            if author and author not in info["author"]:
                return False, f"Author mismatch: expected {info['author']}"
            if year and int(year) != info["year"]:
                return False, f"Year mismatch: expected {info['year']}"
            return True, "Citation verified"
    
    # Not in known database - potentially fabricated
    return False, "Citation not found in known database"


def verify_citations(
    text: str,
    use_api: bool = False,
    api_callback: Callable = None,
) -> CitationVerificationResult:
    """
    Verify citations in generated text.
    
    Args:
        text: Generated text containing citations
        use_api: Whether to use external APIs (Semantic Scholar, etc.)
        api_callback: Optional callback for API verification
        
    Returns:
        CitationVerificationResult with verification analysis
    """
    citations = extract_citations(text)
    
    if not citations:
        return CitationVerificationResult(
            verification_risk=0.0,
            citations_found=[],
            verified_count=0,
            unverified_count=0,
            verification_details=[],
        )
    
    verified_count = 0
    unverified_count = 0
    verification_details = []
    
    for citation in citations:
        # Try local verification first
        verified, reason = verify_citation_local(citation)
        
        if verified:
            verified_count += 1
            verification_details.append(f"✓ {citation}: {reason}")
        else:
            # Could use API verification here
            if use_api and api_callback:
                try:
                    api_verified = api_callback(citation)
                    if api_verified:
                        verified_count += 1
                        verification_details.append(f"✓ {citation}: Verified via API")
                        continue
                except Exception:
                    pass
            
            unverified_count += 1
            verification_details.append(f"✗ {citation}: {reason}")
    
    # Compute risk
    total = verified_count + unverified_count
    if total == 0:
        verification_risk = 0.0
    else:
        unverified_ratio = unverified_count / total
        verification_risk = unverified_ratio * 0.25
    
    return CitationVerificationResult(
        verification_risk=round(verification_risk, 4),
        citations_found=citations,
        verified_count=verified_count,
        unverified_count=unverified_count,
        verification_details=verification_details[:10],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Combined Post-Hoc Analysis
# ─────────────────────────────────────────────────────────────────────────────

def perform_posthoc_analysis(
    response: str,
    prompt: str,
    additional_generations: List[str] = None,
) -> PostHocAnalysis:
    """
    Perform complete post-hoc hallucination analysis.
    
    Args:
        response: Primary generated response
        prompt: Original user prompt
        additional_generations: Optional additional generations for consistency check
        
    Returns:
        PostHocAnalysis with all verification results
    """
    generations = [response]
    if additional_generations:
        generations.extend(additional_generations)
    
    # Run all checks
    consistency = check_self_consistency(generations)
    entropy = compute_semantic_entropy(response, additional_generations)
    verification = verify_citations(response)
    
    # Compute combined risk
    posthoc_risk = (
        consistency.consistency_risk * 0.4 +
        entropy.semantic_entropy * 0.3 +
        verification.verification_risk * 0.3
    )
    posthoc_risk = min(0.50, posthoc_risk)
    
    # Generate summary
    summaries = []
    if consistency.consistency_risk > 0.1:
        summaries.append(f"Inconsistent claims detected ({len(consistency.disagreements)} issues)")
    if entropy.semantic_entropy > 0.1:
        summaries.append(f"High claim diversity ({entropy.num_clusters} clusters)")
    if verification.verification_risk > 0.1:
        summaries.append(f"Unverified citations ({verification.unverified_count}/{len(verification.citations_found)})")
    
    risk_summary = "; ".join(summaries) if summaries else "Post-hoc checks passed"
    
    return PostHocAnalysis(
        posthoc_risk=round(posthoc_risk, 4),
        self_consistency=consistency,
        semantic_entropy=entropy,
        citation_verification=verification,
        risk_summary=risk_summary,
    )
