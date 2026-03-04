"""
Retrieval-based Hallucination Check for HalluScope.

Detects hallucinations by verifying generated claims against
retrievable knowledge. If specific claims (names, dates, facts)
cannot be found in any reference, they may be fabricated.

This is a lightweight implementation without external APIs.
For production, integrate with Wikipedia API or knowledge bases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
from datetime import datetime


@dataclass
class RetrievalAnalysis:
    """Result of retrieval-based verification."""
    
    # Overall retrieval risk [0, 0.25]
    retrieval_risk: float
    
    # Claims that couldn't be verified
    unverified_claims: List[str]
    
    # Risk signals detected
    risk_signals: List[str]
    
    # Breakdown scores
    temporal_risk: float      # Risk from suspicious dates
    citation_risk: float      # Risk from unverifiable citations  
    specificity_risk: float   # Risk from suspiciously specific details
    
    # Risk category
    risk_category: str


# ─────────────────────────────────────────────────────────────────────────────
# Known Facts Registry (Lightweight)
# ─────────────────────────────────────────────────────────────────────────────

# Famous scientists who have VERIFIABLE publications
KNOWN_AUTHORS = {
    "einstein", "newton", "darwin", "curie", "hawking", "feynman", "turing",
    "bohr", "heisenberg", "schrodinger", "maxwell", "planck", "dirac", "lorentz",
    "tesla", "galileo", "copernicus", "kepler", "faraday", "gauss", "euler",
    "pythagoras", "archimedes", "fermat", "riemann", "hilbert", "godel",
    "von neumann", "shannon", "chomsky", "knuth", "dijkstra", "aho",
    "poincare", "noether", "ramanujan", "erdos", "nash", "grothendieck",
    "witten", "penrose", "weinberg", "gell-mann", "higgs", "susskind",
}

# Famous scientists and their domains - used to detect misattribution
KNOWN_SCIENTISTS = {
    "einstein": {"physics", "relativity", "photoelectric", "gravity"},
    "newton": {"physics", "gravity", "calculus", "optics", "motion"},
    "darwin": {"biology", "evolution", "natural selection", "species"},
    "curie": {"physics", "chemistry", "radioactivity", "radium"},
    "hawking": {"physics", "black holes", "cosmology", "time"},
    "feynman": {"physics", "quantum", "qed", "diagrams"},
    "turing": {"computer", "computation", "machine", "cryptography"},
    "bohr": {"physics", "quantum", "atom", "atomic"},
    "heisenberg": {"physics", "quantum", "uncertainty"},
    "schrodinger": {"physics", "quantum", "wave", "equation"},
    "maxwell": {"physics", "electromagnetism", "equations"},
    "planck": {"physics", "quantum", "blackbody", "constant"},
    "lorentz": {"physics", "electromagnetism", "transformation"},
    "dirac": {"physics", "quantum", "electron", "antimatter"},
}

# Major publications and approximate years
KNOWN_PUBLICATIONS = {
    "origin of species": (1859, "darwin"),
    "principia": (1687, "newton"),
    "on the electrodynamics of moving bodies": (1905, "einstein"),
    "brief history of time": (1988, "hawking"),
    "surely you're joking": (1985, "feynman"),
}

# Years when certain concepts were NOT yet discovered
TEMPORAL_ANACHRONISMS = {
    "quantum": 1900,        # Planck's quantum hypothesis
    "relativity": 1905,     # Einstein's special relativity
    "dna": 1953,            # Watson & Crick
    "black hole": 1967,     # Term coined
    "evolution": 1859,      # Origin of Species
    "computer": 1936,       # Turing machine
    "artificial intelligence": 1956,  # Dartmouth conference
    "neural network": 1958,  # Perceptron
}


# ─────────────────────────────────────────────────────────────────────────────
# PRE-GENERATION: Prompt Citation Verification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptCitationVerification:
    """Result of verifying citations in the PROMPT before generation."""
    
    # Overall citation verification risk [0, 0.80]
    citation_risk: float
    
    # Extracted citations from prompt
    citations_found: List[Tuple[str, str, Optional[str]]]  # (title, author, year)
    
    # Verification results
    verified_citations: List[str]      # Citations that could be verified
    unverified_citations: List[str]    # Citations that COULD NOT be verified
    
    # Risk signals
    risk_signals: List[str]
    
    # Is this a citation-requesting prompt?
    is_citation_request: bool


def verify_prompt_citations(prompt: str) -> PromptCitationVerification:
    """
    CRITICAL: Verify citations in the PROMPT BEFORE generation.
    
    This is the key fix for Type-2 hallucinations where the fake entity
    is already present in the prompt. The model will confidently elaborate
    on fake citations unless we detect them BEFORE generation.
    
    Pipeline:
    1. Detect citation patterns in prompt
    2. Extract title, author, year
    3. Verify against known publications
    4. If not found → HIGH RISK
    
    Args:
        prompt: User's input prompt
        
    Returns:
        PromptCitationVerification with risk score and details
    """
    prompt_lower = prompt.lower()
    risk_signals: List[str] = []
    citations_found: List[Tuple[str, str, Optional[str]]] = []
    verified: List[str] = []
    unverified: List[str] = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Detect if this is a citation-requesting prompt
    # ─────────────────────────────────────────────────────────────────────────
    
    citation_indicators = [
        "paper", "article", "publication", "thesis", "dissertation",
        "theorem", "lemma", "corollary", "proof",
        "summarize", "explain", "describe",
    ]
    
    indicator_count = sum(1 for ind in citation_indicators if ind in prompt_lower)
    
    # Check for "Title" by Author pattern (single or double quotes)
    has_title_author = bool(re.search(r'["\'][^"\']{5,}["\'].*\bby\s+[A-Z]', prompt, re.IGNORECASE))
    has_year = bool(re.search(r'\b(18|19|20)\d{2}\b', prompt))
    
    is_citation_request = indicator_count >= 1 and (has_title_author or has_year)
    
    if not is_citation_request:
        # Low confidence this is a citation request
        return PromptCitationVerification(
            citation_risk=0.0,
            citations_found=[],
            verified_citations=[],
            unverified_citations=[],
            risk_signals=[],
            is_citation_request=False,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Extract citations (title, author, year)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Pattern: "Title" by Author or 'Title' by Author
    title_author_patterns = [
        r'["\']([^"\']{5,100})["\'].*?\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'paper\s+["\']([^"\']{5,100})["\']',
        r'paper\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    for pattern in title_author_patterns:
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) >= 2:
                    title, author = match[0], match[1]
                else:
                    title, author = match[0], "Unknown"
            else:
                title, author = match, "Unknown"
            
            # Extract year if present
            year_match = re.search(r'\b((?:18|19|20)\d{2})\b', prompt)
            year = year_match.group(1) if year_match else None
            
            citations_found.append((title.strip(), author.strip(), year))
    
    # Also extract author-only patterns
    author_only = re.findall(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', prompt)
    for author in author_only:
        if not any(author.strip() == c[1] for c in citations_found):
            year_match = re.search(r'\b((?:18|19|20)\d{2})\b', prompt)
            year = year_match.group(1) if year_match else None
            citations_found.append(("(untitled)", author.strip(), year))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Verify each citation
    # ─────────────────────────────────────────────────────────────────────────
    
    citation_risk = 0.0
    
    for title, author, year in citations_found:
        title_lower = title.lower()
        author_lower = author.lower().split()[-1] if author else ""  # Last name
        
        # Check if it's a known publication
        is_known_publication = False
        for known_title, (known_year, known_author) in KNOWN_PUBLICATIONS.items():
            # STRICT matching: require substantial overlap
            # - Exact substring match (known in query OR query in known)
            # - Only match if significant words match (>= 3 chars, not common words)
            common_stopwords = {"the", "a", "an", "of", "in", "on", "by", "for", "to", "and"}
            
            if known_title in title_lower or title_lower in known_title:
                is_known_publication = True
                verified.append(f'"{title}" by {author}')
                break
            
            # Check if key words match (excluding stopwords)
            known_words = set(w for w in known_title.split() if len(w) >= 4 and w not in common_stopwords)
            title_words = set(w for w in title_lower.split() if len(w) >= 4 and w not in common_stopwords)
            
            if known_words and title_words:
                # Require at least 2 significant words to match, or 1 if it's a key unique term
                overlap = known_words & title_words
                if len(overlap) >= 2 or (len(overlap) == 1 and len(list(overlap)[0]) >= 6):
                    is_known_publication = True
                    verified.append(f'"{title}" by {author}')
                    break
        
        # Check if author is known (even if publication isn't)
        is_known_author = author_lower in KNOWN_AUTHORS
        
        if is_known_publication:
            # Verified citation
            risk_signals.append(f"✓ Verified: '{title}' matches known publication")
            continue
        
        # NOT a known publication - this is a hallucination risk!
        
        if not is_known_author:
            # Unknown author + unknown title = HIGH RISK
            citation_risk += 0.40
            risk_signals.append(f"⚠ UNVERIFIED: Author '{author}' not in known database")
            risk_signals.append(f"⚠ UNVERIFIED: Publication '{title}' not found")
            unverified.append(f'"{title}" by {author}')
        else:
            # Known author but unknown publication = MEDIUM RISK
            # Could be a real but obscure paper, or could be fabricated
            citation_risk += 0.20
            risk_signals.append(f"⚠ Unknown publication by known author: '{title}' by {author}")
            unverified.append(f'"{title}" by {author}')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Additional risk signals
    # ─────────────────────────────────────────────────────────────────────────
    
    # Suspiciously specific title patterns
    suspicious_patterns = [
        (r'harmonics\s+in', 0.15, "Harmonics in [X] - common fake pattern"),
        (r'unified\s+theory\s+of', 0.15, "Unified Theory of [X] - grandiose claim"),
        (r'fundamental\s+principles', 0.10, "Fundamental Principles - vague but grand"),
        (r'secret\s+of|hidden\s+truth', 0.15, "Conspiracy-style title"),
    ]
    
    for pattern, risk_add, reason in suspicious_patterns:
        if re.search(pattern, prompt_lower):
            citation_risk += risk_add
            risk_signals.append(f"⚠ {reason}")
    
    # Year-based risk: very old dates less likely to have digital verification
    for title, author, year in citations_found:
        if year:
            year_int = int(year)
            if year_int < 1900:
                citation_risk += 0.10
                risk_signals.append(f"⚠ Historical date ({year}) - harder to verify")
            elif year_int > datetime.now().year:
                citation_risk += 0.25
                risk_signals.append(f"⚠ Future date ({year}) - impossible citation")
    
    # Boost risk if prompt explicitly asks for summary/explanation of unverified work
    if unverified and any(kw in prompt_lower for kw in ["summarize", "explain", "describe"]):
        citation_risk += 0.15
        risk_signals.append("⚠ Requesting summary of unverified publication")
    
    # Cap risk at 0.80
    citation_risk = min(0.80, citation_risk)
    
    return PromptCitationVerification(
        citation_risk=citation_risk,
        citations_found=citations_found,
        verified_citations=verified,
        unverified_citations=unverified,
        risk_signals=risk_signals,
        is_citation_request=is_citation_request,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verification Functions (Post-Generation)
# ─────────────────────────────────────────────────────────────────────────────

def detect_temporal_anachronisms(text: str) -> List[Tuple[str, str]]:
    """
    Detect temporal anachronisms - concepts mentioned before they existed.
    
    Returns:
        List of (claim, reason) tuples
    """
    anachronisms = []
    text_lower = text.lower()
    
    # Find year mentions
    year_matches = re.findall(r'in\s+(\d{4})|(\d{4})[\s,]', text)
    years = [int(y[0] or y[1]) for y in year_matches if y[0] or y[1]]
    
    for year in years:
        if year < 1800 or year > datetime.now().year:
            continue
            
        for concept, discovery_year in TEMPORAL_ANACHRONISMS.items():
            if concept in text_lower and year < discovery_year:
                anachronisms.append((
                    f"'{concept}' mentioned in context of {year}",
                    f"Concept not established until {discovery_year}"
                ))
    
    return anachronisms


def detect_unverifiable_citations(text: str) -> List[Tuple[str, str]]:
    """
    Detect citations that appear suspicious or unverifiable.
    
    Returns:
        List of (claim, reason) tuples
    """
    suspicious = []
    
    # Pattern: "Title" by Author (Year) or (Author, Year)
    citation_pattern = r'"([^"]+)"\s*(?:by|authored by|\()\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    citations = re.findall(citation_pattern, text)
    
    for title, author in citations:
        title_lower = title.lower()
        author_lower = author.lower().split()[-1]  # Get last name
        
        # Check if this is a known publication
        is_known = False
        for known_title, (year, known_author) in KNOWN_PUBLICATIONS.items():
            if known_title in title_lower or title_lower in known_title:
                is_known = True
                break
        
        if not is_known:
            # If not known, flag as potentially fabricated
            suspicious.append((
                f'"{title}" by {author}',
                "Citation cannot be verified against known publications"
            ))
    
    # Detect very specific paper titles that might be fabricated
    specific_title_pattern = r'"([^"]{20,100})"'
    long_titles = re.findall(specific_title_pattern, text)
    
    for title in long_titles:
        # Suspiciously specific and obscure titles
        specificity_signals = [
            "harmonics" in title.lower(),
            "unified theory" in title.lower(),
            "fundamental" in title.lower(),
            bool(re.search(r'\b(?:18|19)\d{2}\b', title)),  # Year in title
        ]
        
        if sum(specificity_signals) >= 2:
            suspicious.append((
                f'"{title}"',
                "Title appears suspiciously specific and potentially fabricated"
            ))
    
    return suspicious


def detect_author_domain_mismatch(text: str) -> List[Tuple[str, str]]:
    """
    Detect when scientists are attributed work outside their known domain.
    
    Returns:
        List of (claim, reason) tuples  
    """
    mismatches = []
    text_lower = text.lower()
    
    for scientist, domains in KNOWN_SCIENTISTS.items():
        if scientist in text_lower:
            # Find what topics are mentioned in same sentence
            sentences = text.split('.')
            for sentence in sentences:
                if scientist in sentence.lower():
                    sentence_lower = sentence.lower()
                    
                    # Check if sentence mentions topics outside scientist's domain
                    mentioned_domains = set()
                    for other_scientist, other_domains in KNOWN_SCIENTISTS.items():
                        if other_scientist != scientist:
                            for domain in other_domains:
                                if domain in sentence_lower:
                                    mentioned_domains.add(domain)
                    
                    # If mentioning domains not in scientist's expertise
                    domain_mismatch = mentioned_domains - domains
                    if domain_mismatch and len(domain_mismatch) > 1:
                        mismatches.append((
                            f"{scientist.title()} attributed work on {', '.join(domain_mismatch)}",
                            f"Known domains are: {', '.join(domains)}"
                        ))
    
    return mismatches


def detect_suspicious_specificity(text: str) -> List[Tuple[str, str]]:
    """
    Detect suspiciously specific details that are hallmark of fabrication.
    
    Returns:
        List of (claim, reason) tuples
    """
    suspicious = []
    
    # Very specific page numbers
    page_pattern = r'page[s]?\s+(\d{3,})'
    pages = re.findall(page_pattern, text.lower())
    for page in pages:
        if int(page) > 500:  # Very specific high page numbers
            suspicious.append((
                f"Page {page} reference",
                "Very specific page number may be fabricated"
            ))
    
    # Specific dates for obscure events
    date_pattern = r'on\s+([A-Z][a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})'
    dates = re.findall(date_pattern, text)
    for month, day, year in dates:
        year_int = int(year)
        if year_int < 1900:  # Very specific historical dates are suspect
            suspicious.append((
                f"Specific date: {month} {day}, {year}",
                "Very specific historical date may be fabricated"
            ))
    
    # Specific measurement values
    measurement_pattern = r'(\d+\.\d{3,})\s*(?:meters?|cm|mm|kg|grams?|seconds?)'
    measurements = re.findall(measurement_pattern, text.lower())
    for measurement in measurements:
        suspicious.append((
            f"Measurement: {measurement}",
            "Very precise measurement without source may be fabricated"
        ))
    
    return suspicious


def verify_response(
    prompt: str,
    response: str,
) -> RetrievalAnalysis:
    """
    Verify a response for potential hallucinations using retrieval heuristics.
    
    Args:
        prompt: Original user prompt
        response: Model's generated response
        
    Returns:
        RetrievalAnalysis with risk scores and signals
    """
    unverified: List[str] = []
    risk_signals: List[str] = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run All Verification Checks
    # ─────────────────────────────────────────────────────────────────────────
    
    # Temporal anachronisms
    temporal_issues = detect_temporal_anachronisms(response)
    temporal_risk = min(0.08, len(temporal_issues) * 0.04)
    for claim, reason in temporal_issues:
        unverified.append(claim)
        risk_signals.append(f"Temporal: {reason}")
    
    # Unverifiable citations
    citation_issues = detect_unverifiable_citations(response)
    citation_risk = min(0.10, len(citation_issues) * 0.05)
    for claim, reason in citation_issues:
        unverified.append(claim)
        risk_signals.append(f"Citation: {reason}")
    
    # Domain mismatches
    domain_issues = detect_author_domain_mismatch(response)
    for claim, reason in domain_issues:
        unverified.append(claim)
        risk_signals.append(f"Domain: {reason}")
        citation_risk = min(0.10, citation_risk + 0.03)
    
    # Suspicious specificity
    specificity_issues = detect_suspicious_specificity(response)
    specificity_risk = min(0.07, len(specificity_issues) * 0.025)
    for claim, reason in specificity_issues:
        unverified.append(claim)
        risk_signals.append(f"Specificity: {reason}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Boost Risk for Known Hallucination Patterns
    # ─────────────────────────────────────────────────────────────────────────
    
    combined_text = (prompt + " " + response).lower()
    
    # Fabricated paper prompt pattern
    if "summarize" in combined_text and ("paper" in combined_text or "article" in combined_text):
        # Check if asking about obscure/fabricated works
        if citation_issues:
            citation_risk = min(0.15, citation_risk + 0.05)
            risk_signals.append("Summarization of unverifiable publication detected")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute Overall Risk
    # ─────────────────────────────────────────────────────────────────────────
    
    retrieval_risk = temporal_risk + citation_risk + specificity_risk
    retrieval_risk = min(0.25, retrieval_risk)
    
    # Determine category
    if retrieval_risk >= 0.12:
        risk_category = "high"
    elif retrieval_risk >= 0.05:
        risk_category = "medium"
    else:
        risk_category = "low"
    
    return RetrievalAnalysis(
        retrieval_risk=round(retrieval_risk, 4),
        unverified_claims=unverified[:10],  # Limit to top 10
        risk_signals=risk_signals[:10],
        temporal_risk=round(temporal_risk, 4),
        citation_risk=round(citation_risk, 4),
        specificity_risk=round(specificity_risk, 4),
        risk_category=risk_category,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick Check (for streaming)
# ─────────────────────────────────────────────────────────────────────────────

def quick_retrieval_check(text: str) -> Tuple[float, List[str]]:
    """
    Quick retrieval check for partial text during streaming.
    
    Returns:
        Tuple of (risk_score, signals)
    """
    analysis = verify_response("", text)
    return analysis.retrieval_risk, analysis.risk_signals
