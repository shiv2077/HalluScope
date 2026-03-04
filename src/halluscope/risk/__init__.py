"""
Risk module for HalluScope.

MULTI-STAGE HALLUCINATION DETECTION ARCHITECTURE:

Stage 1 - Pre-Hoc Analysis (prompt_analysis.py):
    - Citation pattern detection
    - Entity rarity scoring
    - Knowledge cutoff mismatch
    - Compositional novelty

Stage 2 - In-Generation Monitoring (generation_monitor.py):
    - Entity commitment risk
    - Semantic drift detection
    - Real-time HRI updates

Stage 3 - Post-Hoc Verification (post_verification.py):
    - Self-consistency check
    - Semantic entropy computation
    - Citation verification

All signals feed into the HRI (Hallucination Risk Index) computation.
"""

from .hri import (
    sigmoid,
    compute_hri,
    classify_risk,
    compute_hri_with_label,
    explain,
    get_clarification_questions,
    clarification_questions_high,
    clarification_questions_medium,
    # Legacy
    compute_hri_v1,
    explain_v1,
    # Constants
    DEFAULT_WEIGHTS,
    THRESHOLD_LOW,
    THRESHOLD_HIGH,
)

# STAGE 1: Pre-Hoc Analysis
from .prompt_analysis import (
    compute_prompt_risk,
    PromptRiskAnalysis,
)

# Entity analysis (used across stages)
from .entity_analysis import (
    extract_entities,
    compute_entity_novelty,
    EntityTracker,
    EntityAnalysis,
)

# Retrieval check (legacy, being replaced)
from .retrieval_check import (
    verify_response,
    quick_retrieval_check,
    RetrievalAnalysis,
    # NEW: Pre-generation citation verification
    verify_prompt_citations,
    PromptCitationVerification,
)

# STAGE 2: In-Generation Monitoring
from .generation_monitor import (
    GenerationMonitor,
    EntityCommitmentResult,
    SemanticDriftResult,
    compute_entity_commitment_risk,
    compute_semantic_drift,
)

# STAGE 3: Post-Hoc Verification
from .post_verification import (
    perform_posthoc_analysis,
    check_self_consistency,
    compute_semantic_entropy,
    verify_citations,
    PostHocAnalysis,
    SelfConsistencyResult,
    SemanticEntropyResult,
    CitationVerificationResult,
)

__all__ = [
    # HRI core
    "sigmoid",
    "compute_hri",
    "classify_risk",
    "compute_hri_with_label",
    "explain",
    "get_clarification_questions",
    "clarification_questions_high",
    "clarification_questions_medium",
    "compute_hri_v1",
    "explain_v1",
    "DEFAULT_WEIGHTS",
    "THRESHOLD_LOW",
    "THRESHOLD_HIGH",
    # Stage 1: Prompt analysis
    "compute_prompt_risk",
    "PromptRiskAnalysis",
    # Entity analysis
    "extract_entities",
    "compute_entity_novelty",
    "EntityTracker",
    "EntityAnalysis",
    # Retrieval check (legacy + pre-gen citation verification)
    "verify_response",
    "quick_retrieval_check",
    "RetrievalAnalysis",
    "verify_prompt_citations",
    "PromptCitationVerification",
    # Stage 2: Generation monitoring
    "GenerationMonitor",
    "EntityCommitmentResult",
    "SemanticDriftResult",
    "compute_entity_commitment_risk",
    "compute_semantic_drift",
    # Stage 3: Post-hoc verification
    "perform_posthoc_analysis",
    "check_self_consistency",
    "compute_semantic_entropy",
    "verify_citations",
    "PostHocAnalysis",
    "SelfConsistencyResult",
    "SemanticEntropyResult",
    "CitationVerificationResult",
]
