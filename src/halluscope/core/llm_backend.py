"""
LLM backend for HalluScope.

Loads an open-source causal LLM via HuggingFace Transformers and provides
generation with REAL-TIME hallucination risk monitoring.

MULTI-STAGE HALLUCINATION DETECTION ARCHITECTURE:

Stage 1 - PRE-HOC (Before Generation):
    - Citation pattern detection
    - Entity rarity scoring
    - Knowledge cutoff mismatch
    - Compositional novelty
    
Stage 2 - IN-GENERATION (During Token Generation):
    - Token entropy & logprob variance
    - Entity commitment risk
    - Semantic drift detection
    - Real-time HRI updates
    
Stage 3 - POST-HOC (After Generation):
    - Self-consistency check
    - Semantic entropy
    - Citation verification

TokenBurn-style: Pre-hoc detection with manual autoregressive decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Stage 1: Pre-hoc analysis
from ..risk.prompt_analysis import compute_prompt_risk, PromptRiskAnalysis

# Import Entity analysis (shared)
from ..risk.entity_analysis import EntityTracker, compute_entity_novelty, EntityAnalysis

# Import legacy retrieval check
from ..risk.retrieval_check import (
    verify_response, 
    quick_retrieval_check, 
    RetrievalAnalysis,
    verify_prompt_citations,
    PromptCitationVerification,
)

# Import Stage 2: In-generation monitoring
from ..risk.generation_monitor import (
    GenerationMonitor, 
    EntityCommitmentResult, 
    SemanticDriftResult,
)

# Import Stage 3: Post-hoc verification
from ..risk.post_verification import (
    perform_posthoc_analysis,
    PostHocAnalysis,
)


# ─────────────────────────────────────────────────────────────────────────────
# HRI Configuration - MULTI-STAGE FORMULA
# ─────────────────────────────────────────────────────────────────────────────

HRI_WEIGHTS = {
    # Stage 1: Pre-hoc signals (prompt analysis)
    "wP": 0.10,   # Prompt risk weight (citation + specificity + summary)
    "wER": 0.06,  # Entity rarity weight
    "wCO": 0.06,  # Cutoff risk weight (knowledge cutoff mismatch)
    "wCN": 0.04,  # Compositional novelty weight
    
    # Stage 2: In-generation signals (uncertainty)
    "wH": 0.10,   # Entropy weight
    "wN": 0.08,   # NLL weight
    "wV": 0.04,   # Logprob std weight
    "wEC": 0.06,  # Entity commitment risk weight
    "wD": 0.04,   # Semantic drift weight
    
    # Stage 3: Post-hoc signals (verification) - BOOSTED WEIGHTS
    "wE": 0.06,   # Entity novelty weight
    "wR": 0.20,   # Retrieval/citation verification risk weight - CRITICAL!
    "wSC": 0.04,  # Self-consistency risk weight
    "wSE": 0.03,  # Semantic entropy weight
    "wCV": 0.08,  # Citation verification risk weight (post-gen)
    
    # Bias - slightly reduced to allow pre-gen signals to trigger
    "b": -1.9,    # Bias term (adjusted for citation verification gate)
}

# Thresholds
HRI_THRESHOLD = 0.65          # Early stopping threshold
HRI_INTERVENTION_THRESHOLD = 0.55  # Real-time intervention threshold
TOP_K_ENTROPY = 50            # Use top-k for fast entropy computation


@dataclass
class GenerationResult:
    """Result of LLM generation with multi-stage hallucination risk analysis."""
    
    # Core output
    text: str
    prompt: str
    
    # Token-level data for visualization
    tokens: List[str]              # Decoded token strings
    token_risks: List[float]       # Per-token risk scores [0, 1]
    
    # HRI trajectory (one value per generated token)
    hri_values: List[float]        # Running HRI after each token
    
    # Real-time intervention
    early_stopped: bool            # True if generation halted due to HRI threshold
    intervention_triggered: bool   # True if HRI exceeded intervention threshold
    clarification_questions: Optional[List[str]] = None  # Questions if high risk
    intervention_message: Optional[str] = None  # Clarification prompt
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: PRE-HOC ANALYSIS (Prompt Risk)
    # ─────────────────────────────────────────────────────────────────────────
    
    prompt_risk: float = 0.0                  # Combined prompt risk [0, 0.60]
    prompt_analysis: Optional[PromptRiskAnalysis] = None
    
    # Stage 1 component signals
    citation_risk: float = 0.0                # Citation pattern risk [0, 0.15]
    entity_rarity: float = 0.0                # Entity rarity score [0, 0.15]
    cutoff_risk: float = 0.0                  # Knowledge cutoff risk [0, 0.20]
    compositional_novelty: float = 0.0        # Compositional novelty [0, 0.15]
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: IN-GENERATION SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    
    # Entity analysis (during generation)
    entity_risk: float = 0.0                  # Novel entity risk [0, 0.25]
    entity_analysis: Optional[EntityAnalysis] = None
    novel_entities: List[str] = field(default_factory=list)
    
    # Entity commitment (Stage 2)
    entity_commitment_risk: float = 0.0       # Risk from committing to entities
    commitment_result: Optional[EntityCommitmentResult] = None
    
    # Semantic drift (Stage 2)
    semantic_drift: float = 0.0               # Drift from prompt [0, 0.25]
    drift_result: Optional[SemanticDriftResult] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: POST-HOC VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    
    # Retrieval check
    retrieval_risk: float = 0.0               # Unverifiable claims risk [0, 0.25]
    retrieval_analysis: Optional[RetrievalAnalysis] = None
    
    # Self-consistency (requires multiple generations)
    self_consistency_risk: float = 0.0        # Inconsistency risk [0, 0.25]
    
    # Semantic entropy
    semantic_entropy: float = 0.0             # Claim entropy [0, 0.25]
    
    # Citation verification
    citation_verification_risk: float = 0.0   # Unverified citations [0, 0.25]
    
    # Complete post-hoc analysis
    posthoc_analysis: Optional[PostHocAnalysis] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # RISK BREAKDOWN (for UI/debugging)
    # ─────────────────────────────────────────────────────────────────────────
    
    risk_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Risk slope detection (ΔHRI/Δtokens) - rapid increase signals trouble
    hri_slope: float = 0.0                    # Rate of HRI change
    hri_acceleration: float = 0.0             # 2nd derivative - spikes
    
    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate metrics (for backward compatibility)
    # ─────────────────────────────────────────────────────────────────────────
    
    entropy_mean: float = 0.0
    entropy_max: float = 0.0
    nll: float = 0.0
    perplexity: float = 1.0
    logprob_std: float = 0.0
    logprob_variance: float = 0.0
    
    # Token IDs (for advanced use)
    input_ids: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    new_token_ids: Optional[torch.Tensor] = None
    vocab_size: int = 0
    
    # Legacy fields (deprecated, kept for compatibility)
    scores: Optional[List[torch.Tensor]] = None
    topk_token_ids: Optional[List[torch.Tensor]] = None
    topk_probs: Optional[List[torch.Tensor]] = None


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


class LLMBackend:
    """
    LLM backend with REAL-TIME hallucination risk monitoring.
    
    Uses manual autoregressive decoding (not model.generate()) to:
    - Compute entropy and log-prob at each token
    - Maintain running HRI estimate
    - Enable early stopping when HRI exceeds threshold
    - Return token-level risk scores for UI highlighting
    """
    
    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id

        # GPU detection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ Running HalluScope model on: {gpu_name}")
            else:
                self.device = "cpu"
                print("⚠️  No GPU detected. Running HalluScope model on: CPU (slow)")
        else:
            self.device = device
            print(f"Running HalluScope model on: {device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = self.tokenizer.vocab_size

        # Load model with GPU acceleration
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading model: {model_id} (dtype={dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        ).eval()
        print(f"✅ Model loaded successfully on {self.model.device}")

    def _compute_token_entropy_topk(self, probs: torch.Tensor, k: int = TOP_K_ENTROPY) -> float:
        """
        Compute entropy using top-k probabilities for speed.
        
        Args:
            probs: Probability distribution [vocab_size]
            k: Number of top probabilities to use
            
        Returns:
            Approximate entropy in nats
        """
        try:
            top_probs, _ = torch.topk(probs, min(k, probs.size(-1)))
            # Clamp to avoid log(0)
            top_probs = torch.clamp(top_probs, min=1e-12, max=1.0)
            # Entropy: H = -sum(p * log(p))
            entropy = -(top_probs * torch.log(top_probs)).sum().item()
            # Handle NaN/Inf
            if math.isnan(entropy) or math.isinf(entropy):
                return 3.0  # Default moderate entropy
            return entropy
        except Exception:
            return 3.0  # Default moderate entropy on error
    
    def _compute_token_risk(self, entropy: float, logprob: float) -> float:
        """
        Compute per-token risk score.
        
        Formula: token_risk = sigmoid(0.5 * entropy + 0.5 * (-logprob))
        
        Args:
            entropy: Token entropy (higher = more uncertain)
            logprob: Log probability of token (negative value)
            
        Returns:
            Risk score in [0, 1]
        """
        # Handle NaN/Inf inputs
        if math.isnan(entropy) or math.isinf(entropy):
            entropy = 3.0
        if math.isnan(logprob) or math.isinf(logprob):
            logprob = -5.0
        
        # Scale entropy roughly to [0, 5] range for sigmoid
        scaled_entropy = entropy / 2.0  # Typical entropy ~2-5 nats
        # -logprob is positive; typical range 0-10
        scaled_nll = -logprob / 3.0
        
        x = 0.5 * scaled_entropy + 0.5 * scaled_nll - 1.5
        result = _sigmoid(x)
        
        # Final NaN check
        if math.isnan(result):
            return 0.5
        return result
    
    def _compute_running_hri(
        self,
        entropies: List[float],
        logprobs: List[float],
        # Stage 1: Pre-hoc signals
        prompt_risk: float = 0.0,
        entity_rarity: float = 0.0,
        cutoff_risk: float = 0.0,
        compositional_novelty: float = 0.0,
        # Stage 2: In-generation signals
        entity_commitment_risk: float = 0.0,
        semantic_drift: float = 0.0,
        # Stage 3 / Legacy signals
        entity_risk: float = 0.0,
        retrieval_risk: float = 0.0,
        self_consistency_risk: float = 0.0,
        semantic_entropy: float = 0.0,
        citation_verification_risk: float = 0.0,
        debug: bool = False,
    ) -> float:
        """
        Compute running HRI from accumulated metrics using MULTI-STAGE formula.
        
        Formula: HRI = sigmoid(
            # Stage 1 (Pre-hoc)
            wP*prompt_risk + wER*entity_rarity + wCO*cutoff + wCN*comp_novelty +
            # Stage 2 (In-generation) 
            wH*entropy + wN*nll + wV*std + wEC*entity_commitment + wD*drift +
            # Stage 3 (Post-hoc)
            wE*entity + wR*retrieval + wSC*consistency + wSE*sem_entropy + wCV*citation +
            b
        )
        
        Returns:
            HRI score in [0, 1]
        """
        if not entropies or not logprobs:
            # If no uncertainty data yet, rely on pre-generation signals
            x = (
                HRI_WEIGHTS["wP"] * prompt_risk * 10 +
                HRI_WEIGHTS.get("wER", 0.08) * entity_rarity * 10 +
                HRI_WEIGHTS.get("wCO", 0.08) * cutoff_risk * 10 +
                HRI_WEIGHTS["b"]
            )
            return _sigmoid(x)
        
        # Filter out NaN/Inf values
        valid_entropies = [e for e in entropies if not (math.isnan(e) or math.isinf(e))]
        valid_logprobs = [lp for lp in logprobs if not (math.isnan(lp) or math.isinf(lp))]
        
        if not valid_entropies:
            valid_entropies = [3.0]  # Default moderate entropy
        if not valid_logprobs:
            valid_logprobs = [-5.0]  # Default moderate logprob
        
        # Mean entropy
        entropy_mean = sum(valid_entropies) / len(valid_entropies)
        
        # Mean NLL (negate logprobs)
        nll_mean = -sum(valid_logprobs) / len(valid_logprobs)
        
        # Logprob standard deviation (instability signal)
        if len(valid_logprobs) >= 2:
            mean_lp = sum(valid_logprobs) / len(valid_logprobs)
            variance = sum((lp - mean_lp) ** 2 for lp in valid_logprobs) / len(valid_logprobs)
            std = math.sqrt(max(0, variance))  # Ensure non-negative
        else:
            std = 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # MULTI-STAGE HRI FORMULA
        # ─────────────────────────────────────────────────────────────────────
        
        # Stage 1: Pre-hoc components
        comp_prompt = HRI_WEIGHTS["wP"] * prompt_risk * 10
        comp_entity_rarity = HRI_WEIGHTS.get("wER", 0.08) * entity_rarity * 10
        comp_cutoff = HRI_WEIGHTS.get("wCO", 0.08) * cutoff_risk * 10
        comp_novelty = HRI_WEIGHTS.get("wCN", 0.06) * compositional_novelty * 10
        
        # Stage 2: In-generation components
        comp_entropy = HRI_WEIGHTS["wH"] * entropy_mean
        comp_nll = HRI_WEIGHTS["wN"] * nll_mean
        comp_std = HRI_WEIGHTS["wV"] * std
        comp_commitment = HRI_WEIGHTS.get("wEC", 0.08) * entity_commitment_risk * 10
        comp_drift = HRI_WEIGHTS.get("wD", 0.06) * semantic_drift * 10
        
        # Stage 3: Post-hoc components
        comp_entity = HRI_WEIGHTS.get("wE", 0.08) * entity_risk * 10
        comp_retrieval = HRI_WEIGHTS.get("wR", 0.06) * retrieval_risk * 10
        comp_consistency = HRI_WEIGHTS.get("wSC", 0.05) * self_consistency_risk * 10
        comp_sem_entropy = HRI_WEIGHTS.get("wSE", 0.04) * semantic_entropy * 10
        comp_citation = HRI_WEIGHTS.get("wCV", 0.05) * citation_verification_risk * 10
        
        # Bias
        comp_bias = HRI_WEIGHTS["b"]
        
        # Sum all components
        x = (
            comp_prompt + comp_entity_rarity + comp_cutoff + comp_novelty +  # Stage 1
            comp_entropy + comp_nll + comp_std + comp_commitment + comp_drift +  # Stage 2
            comp_entity + comp_retrieval + comp_consistency + comp_sem_entropy + comp_citation +  # Stage 3
            comp_bias
        )
        
        result = _sigmoid(x)
        
        # DEBUG: Print full multi-stage signal breakdown
        if debug:
            print("\n" + "="*70)
            print("=== HRI MULTI-STAGE SIGNAL BREAKDOWN ===")
            print("="*70)
            print("\n📋 STAGE 1: PRE-HOC ANALYSIS (Before Generation)")
            print("-" * 50)
            print(f"  prompt_risk:           {prompt_risk:.4f} → weighted: {comp_prompt:.4f}")
            print(f"  entity_rarity:         {entity_rarity:.4f} → weighted: {comp_entity_rarity:.4f}")
            print(f"  cutoff_risk:           {cutoff_risk:.4f} → weighted: {comp_cutoff:.4f}")
            print(f"  compositional_novelty: {compositional_novelty:.4f} → weighted: {comp_novelty:.4f}")
            
            print("\n⚡ STAGE 2: IN-GENERATION MONITORING")
            print("-" * 50)
            print(f"  entropy_mean:          {entropy_mean:.4f} → weighted: {comp_entropy:.4f}")
            print(f"  nll_mean:              {nll_mean:.4f} → weighted: {comp_nll:.4f}")
            print(f"  logprob_std:           {std:.4f} → weighted: {comp_std:.4f}")
            print(f"  entity_commitment:     {entity_commitment_risk:.4f} → weighted: {comp_commitment:.4f}")
            print(f"  semantic_drift:        {semantic_drift:.4f} → weighted: {comp_drift:.4f}")
            
            print("\n🔍 STAGE 3: POST-HOC VERIFICATION")
            print("-" * 50)
            print(f"  entity_novelty:        {entity_risk:.4f} → weighted: {comp_entity:.4f}")
            print(f"  retrieval_risk:        {retrieval_risk:.4f} → weighted: {comp_retrieval:.4f}")
            print(f"  self_consistency:      {self_consistency_risk:.4f} → weighted: {comp_consistency:.4f}")
            print(f"  semantic_entropy:      {semantic_entropy:.4f} → weighted: {comp_sem_entropy:.4f}")
            print(f"  citation_verification: {citation_verification_risk:.4f} → weighted: {comp_citation:.4f}")
            
            print("\n📊 FINAL COMPUTATION")
            print("-" * 50)
            print(f"  bias:         {comp_bias:.4f}")
            print(f"  raw_score:    {x:.4f}")
            print(f"  sigmoid(x):   {result:.4f}")
            print(f"  HRI:          {result:.4f}")
            
            # Risk classification
            if result >= 0.65:
                risk_label = "🔴 HIGH"
            elif result >= 0.45:
                risk_label = "🟡 MEDIUM"
            else:
                risk_label = "🟢 LOW"
            print(f"  Risk Level:   {risk_label}")
            print("="*70 + "\n")
        
        # Final NaN check
        if math.isnan(result) or math.isinf(result):
            return 0.3  # Fallback to moderate-low risk
        
        return result
    
    def _compute_hri_slope(self, hri_values: List[float], window: int = 5) -> Tuple[float, float]:
        """
        Compute HRI slope (rate of change) and acceleration.
        
        Rapid increases in HRI signal emerging hallucination.
        
        Args:
            hri_values: List of HRI values over generation
            window: Window size for slope computation
            
        Returns:
            Tuple of (slope, acceleration)
        """
        if len(hri_values) < 3:
            return 0.0, 0.0
        
        # Use last N values for recent trend
        recent = hri_values[-min(window, len(hri_values)):]
        
        # Compute slope: (last - first) / num_steps
        slope = (recent[-1] - recent[0]) / len(recent)
        
        # Compute acceleration (change in slope)
        if len(hri_values) >= 6:
            earlier = hri_values[-6:-3] if len(hri_values) >= 6 else hri_values[:3]
            earlier_slope = (earlier[-1] - earlier[0]) / len(earlier) if len(earlier) > 1 else 0.0
            acceleration = slope - earlier_slope
        else:
            acceleration = 0.0
        
        return slope, acceleration
    
    def _get_clarification_questions(self) -> List[str]:
        """Return clarification questions for high-risk generations."""
        return [
            "Can you specify the exact entity, date, or location the answer must be grounded on?",
            "Do you want a strictly factual answer (must be verifiable) or a conceptual explanation?",
            "What source or context should the response rely on (notes, paper, dataset, etc.)?",
        ]
    
    def _generate_intervention_message(
        self,
        prompt: str,
        prompt_analysis: PromptRiskAnalysis,
        current_hri: float,
    ) -> str:
        """
        Generate a real-time intervention message when HRI exceeds threshold.
        
        The message explains what the system detected and asks for clarification.
        """
        messages = []
        
        # Header
        messages.append(f"⚠️ High hallucination risk detected (HRI: {current_hri:.2f})")
        messages.append("")
        
        # Explain detected risks
        if prompt_analysis.unverifiable_risk > 0.05:
            # Extract title/author if possible
            import re
            title_match = re.search(r'"([^"]+)"', prompt)
            author_match = re.search(r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', prompt)
            
            if title_match and author_match:
                messages.append(f"I cannot verify the existence of the paper")
                messages.append(f'"{title_match.group(1)}" by {author_match.group(1)}.')
            else:
                messages.append("I cannot verify the specific citation or publication referenced.")
        
        if prompt_analysis.cutoff_risk > 0.05:
            messages.append("The query references dates that may be beyond my knowledge cutoff.")
        
        if prompt_analysis.entity_rarity > 0.08:
            messages.append("I cannot verify some entities mentioned in the query.")
        
        if prompt_analysis.compositional_novelty > 0.08:
            messages.append("The combination of concepts in this query is unusual.")
        
        messages.append("")
        messages.append("Do you have a source or additional context?")
        
        return "\n".join(messages)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: Optional[bool] = None,
        hri_threshold: float = HRI_THRESHOLD,
        early_stop_enabled: bool = True,
        # Legacy parameters (ignored but kept for compatibility)
        top_k_probs: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate text with REAL-TIME hallucination risk monitoring.
        
        Uses manual autoregressive decoding to compute uncertainty metrics
        at each token, enabling early stopping when HRI exceeds threshold.
        
        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum tokens to generate (default 60 for speed).
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling constraint.
            do_sample: Whether to sample (default: True if temperature > 0).
            hri_threshold: HRI value that triggers early stopping (default 0.70).
            early_stop_enabled: Whether to enable early stopping.
        
        Returns:
            GenerationResult with tokens, token_risks, hri_values, and metrics.
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: PRE-GENERATION - Analyze prompt for hallucination risk patterns
        # ─────────────────────────────────────────────────────────────────────
        
        prompt_analysis_result = compute_prompt_risk(prompt)
        prompt_risk = prompt_analysis_result.risk_score
        
        # Extract Stage 1 component signals
        entity_rarity = prompt_analysis_result.entity_rarity
        cutoff_risk = prompt_analysis_result.cutoff_risk
        compositional_novelty = prompt_analysis_result.compositional_novelty
        
        # ─────────────────────────────────────────────────────────────────────
        # CRITICAL: Pre-generation citation verification (fixes Type-2 hallucinations)
        # ─────────────────────────────────────────────────────────────────────
        
        citation_verification = verify_prompt_citations(prompt)
        
        # If prompt contains unverifiable citations, set retrieval_risk BEFORE generation
        # This is the key fix: citation risk should be high BEFORE model starts generating
        initial_retrieval_risk = citation_verification.citation_risk
        
        # Boost prompt_risk based on citation verification
        if citation_verification.unverified_citations:
            # Add citation risk directly to prompt risk (this needs to flow into HRI)
            prompt_risk = min(0.80, prompt_risk + citation_verification.citation_risk * 0.5)
            
            # Log for debugging
            for signal in citation_verification.risk_signals:
                if signal.startswith("⚠"):
                    print(f"[PRE-GEN] {signal}")
        
        # Initialize entity tracker with entities from prompt
        entity_tracker = EntityTracker(prompt_analysis_result.prompt_entities)
        
        # Initialize Stage 2: Generation Monitor
        generation_monitor = GenerationMonitor(prompt, prompt_analysis_result.prompt_entities)
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]
        
        # Storage for generated tokens and metrics
        generated_tokens: List[int] = []
        token_strings: List[str] = []
        token_entropies: List[float] = []
        token_logprobs: List[float] = []
        token_risks: List[float] = []
        hri_values: List[float] = []
        
        early_stopped = False
        intervention_triggered = False
        intervention_message = None
        current_ids = input_ids
        
        # Running risk accumulators - initialized from pre-generation analysis
        running_entity_risk = 0.0
        running_retrieval_risk = initial_retrieval_risk  # START with citation verification risk!
        running_commitment_risk = 0.0
        running_drift = 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # Check for high-risk BEFORE generation starts
        # ─────────────────────────────────────────────────────────────────────
        
        # Compute pre-generation HRI with citation risk
        pre_gen_hri = self._compute_running_hri(
            entropies=[],  # No tokens yet
            logprobs=[],
            prompt_risk=prompt_risk,
            entity_rarity=entity_rarity,
            cutoff_risk=cutoff_risk,
            compositional_novelty=compositional_novelty,
            retrieval_risk=running_retrieval_risk,
        )
        
        # If pre-generation HRI is already high, trigger intervention immediately
        if pre_gen_hri >= HRI_INTERVENTION_THRESHOLD:
            intervention_triggered = True
            intervention_message = self._generate_intervention_message(
                prompt, prompt_analysis_result, pre_gen_hri
            )
            # Add specific citation warning
            if citation_verification.unverified_citations:
                citations_str = ", ".join(citation_verification.unverified_citations)
                intervention_message = f"⚠️ Cannot verify citation(s): {citations_str}\n\n" + intervention_message
        
        # ─────────────────────────────────────────────────────────────────────
        # Manual Autoregressive Decoding Loop with MULTI-SIGNAL monitoring
        # ─────────────────────────────────────────────────────────────────────
        
        for step in range(max_new_tokens):
            # Forward pass
            outputs = self.model(input_ids=current_ids)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Compute probabilities
            probs = F.softmax(logits[0], dim=-1)  # [vocab_size]
            
            # ───────────────────────────────────────────────────────────────
            # Sample next token
            # ───────────────────────────────────────────────────────────────
            
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    # Renormalize
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    
                    # Top-p (nucleus) filtering on top-k
                    sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum <= top_p
                    # Keep at least one token
                    mask[0] = True
                    filtered_probs = sorted_probs * mask
                    filtered_probs = filtered_probs / filtered_probs.sum()
                    
                    # Sample from filtered distribution
                    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
                    local_idx = sorted_indices[sample_idx]
                    next_token = top_k_indices[local_idx].item()
                else:
                    # Just sample from full distribution
                    next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding
                next_token = torch.argmax(probs).item()
            
            # ───────────────────────────────────────────────────────────────
            # Compute uncertainty metrics for this token
            # ───────────────────────────────────────────────────────────────
            
            # Token entropy (top-k approximation for speed)
            entropy = self._compute_token_entropy_topk(probs, TOP_K_ENTROPY)
            
            # Token log probability
            logprob = torch.log(probs[next_token] + 1e-12).item()
            
            # Token risk score
            token_risk = self._compute_token_risk(entropy, logprob)
            
            # Store metrics
            token_entropies.append(entropy)
            token_logprobs.append(logprob)
            token_risks.append(token_risk)
            generated_tokens.append(next_token)
            
            # Decode token to string
            token_str = self.tokenizer.decode([next_token])
            token_strings.append(token_str)
            
            # ───────────────────────────────────────────────────────────────
            # ENTITY TRACKING: Update entity novelty risk
            # ───────────────────────────────────────────────────────────────
            
            # Use add_token method with all required parameters
            entity_risk_increment = entity_tracker.add_token(
                token_str, 
                token_strings, 
                len(token_strings) - 1
            )
            running_entity_risk = min(0.25, running_entity_risk + entity_risk_increment)
            
            # ───────────────────────────────────────────────────────────────
            # STAGE 2: In-Generation Monitoring
            # ───────────────────────────────────────────────────────────────
            
            # Update generation monitor with new token
            stage2_signals = generation_monitor.add_token(token_str, entropy, logprob)
            running_commitment_risk = stage2_signals["entity_commitment_risk"]
            running_drift = stage2_signals["semantic_drift"]
            
            # Periodically compute retrieval risk on partial text
            current_text = "".join(token_strings)
            if step > 0 and step % 10 == 0 and len(current_text) > 20:
                partial_retrieval_risk, _ = quick_retrieval_check(current_text)
                running_retrieval_risk = partial_retrieval_risk
            
            # ───────────────────────────────────────────────────────────────
            # Compute running HRI with MULTI-STAGE formula
            # ───────────────────────────────────────────────────────────────
            
            current_hri = self._compute_running_hri(
                token_entropies, 
                token_logprobs,
                # Stage 1
                prompt_risk=prompt_risk,
                entity_rarity=entity_rarity,
                cutoff_risk=cutoff_risk,
                compositional_novelty=compositional_novelty,
                # Stage 2
                entity_commitment_risk=running_commitment_risk,
                semantic_drift=running_drift,
                # Stage 3 (partial)
                entity_risk=running_entity_risk,
                retrieval_risk=running_retrieval_risk,
            )
            hri_values.append(current_hri)
            
            # ───────────────────────────────────────────────────────────────
            # REAL-TIME INTERVENTION CHECK
            # ───────────────────────────────────────────────────────────────
            
            if not intervention_triggered and current_hri >= HRI_INTERVENTION_THRESHOLD:
                intervention_triggered = True
                # Generate intervention message based on detected risks
                intervention_message = self._generate_intervention_message(
                    prompt, prompt_analysis_result, current_hri
                )
            
            # ───────────────────────────────────────────────────────────────
            # Early stopping check (PRE-HOC DETECTION)
            # ───────────────────────────────────────────────────────────────
            
            if early_stop_enabled and current_hri >= hri_threshold:
                early_stopped = True
                break
            
            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Append token and continue
            next_token_tensor = torch.tensor([[next_token]], device=self.model.device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=-1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Compute aggregate metrics
        # ─────────────────────────────────────────────────────────────────────
        
        # Filter out NaN/Inf values for aggregate calculations
        valid_entropies = [e for e in token_entropies if not (math.isnan(e) or math.isinf(e))]
        valid_logprobs = [lp for lp in token_logprobs if not (math.isnan(lp) or math.isinf(lp))]
        valid_risks = [r for r in token_risks if not (math.isnan(r) or math.isinf(r))]
        valid_hris = [h for h in hri_values if not (math.isnan(h) or math.isinf(h))]
        
        if valid_entropies:
            entropy_mean = sum(valid_entropies) / len(valid_entropies)
            entropy_max = max(valid_entropies)
        else:
            entropy_mean = 3.0  # Default moderate
            entropy_max = 3.0
        
        if valid_logprobs:
            nll = -sum(valid_logprobs) / len(valid_logprobs)
            nll = max(0.0, min(nll, 100.0))  # Clamp to reasonable range
            ppl = math.exp(min(nll, 100))  # Cap to prevent overflow
            
            if len(valid_logprobs) >= 2:
                mean_lp = sum(valid_logprobs) / len(valid_logprobs)
                variance = sum((lp - mean_lp) ** 2 for lp in valid_logprobs) / len(valid_logprobs)
                std = math.sqrt(max(0, variance))
            else:
                variance = 0.0
                std = 0.0
        else:
            nll = 2.0  # Default moderate
            ppl = math.exp(2.0)
            variance = 0.0
            std = 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # Build output
        # ─────────────────────────────────────────────────────────────────────
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        
        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: POST-GENERATION - Complete hallucination verification
        # ─────────────────────────────────────────────────────────────────────
        
        # Get Stage 2 final results
        commitment_result = generation_monitor.get_commitment_result()
        drift_result = generation_monitor.get_drift_result()
        final_commitment_risk = commitment_result.commitment_risk
        final_drift = drift_result.drift_score
        
        # Final entity analysis
        final_entity_analysis = compute_entity_novelty(prompt, generated_text)
        final_entity_risk = final_entity_analysis.novelty_score
        
        # Full retrieval verification
        final_retrieval_analysis = verify_response(prompt, generated_text)
        final_retrieval_risk = final_retrieval_analysis.retrieval_risk
        
        # Stage 3: Post-hoc analysis (self-consistency, semantic entropy, citation verification)
        # Note: Self-consistency requires multiple generations - we only have one here
        posthoc_analysis = perform_posthoc_analysis(
            response=generated_text,
            prompt=prompt,
            additional_generations=None,  # Would need multiple generations
        )
        
        # Extract Stage 3 signals
        self_consistency_risk = posthoc_analysis.self_consistency.consistency_risk
        semantic_entropy_risk = posthoc_analysis.semantic_entropy.semantic_entropy
        citation_verification_risk = posthoc_analysis.citation_verification.verification_risk
        
        # Use the higher of running or final risk values
        final_entity_risk = max(running_entity_risk, final_entity_risk)
        final_retrieval_risk = max(running_retrieval_risk, final_retrieval_risk)
        
        # Build comprehensive risk breakdown for UI visualization
        risk_breakdown = {
            # Stage 1: Pre-hoc
            "prompt_risk": prompt_risk,
            "citation_risk": prompt_analysis_result.citation_risk,
            "entity_rarity": entity_rarity,
            "cutoff_risk": cutoff_risk,
            "compositional_novelty": compositional_novelty,
            # Stage 2: In-generation
            "entropy": entropy_mean,
            "nll": nll,
            "logprob_std": std,
            "entity_commitment_risk": final_commitment_risk,
            "semantic_drift": final_drift,
            # Stage 3: Post-hoc
            "entity_risk": final_entity_risk,
            "retrieval_risk": final_retrieval_risk,
            "self_consistency_risk": self_consistency_risk,
            "semantic_entropy": semantic_entropy_risk,
            "citation_verification_risk": citation_verification_risk,
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # FINAL HRI: Recompute with ALL multi-stage signals
        # ─────────────────────────────────────────────────────────────────────
        
        # DEBUG: Print comprehensive stage analysis
        print("\n" + "="*70)
        print("=== MULTI-STAGE HALLUCINATION DETECTION RESULTS ===")
        print("="*70)
        
        print("\n📋 STAGE 1: PRE-HOC ANALYSIS")
        print("-" * 50)
        print(f"  prompt_risk:           {prompt_risk:.4f}")
        print(f"    citation_risk:       {prompt_analysis_result.citation_risk:.4f}")
        print(f"    specificity_risk:    {prompt_analysis_result.specificity_risk:.4f}")
        print(f"    summary_risk:        {prompt_analysis_result.summary_risk:.4f}")
        print(f"    unverifiable_risk:   {prompt_analysis_result.unverifiable_risk:.4f}")
        print(f"  entity_rarity:         {entity_rarity:.4f}")
        print(f"  cutoff_risk:           {cutoff_risk:.4f}")
        print(f"  compositional_novelty: {compositional_novelty:.4f}")
        print(f"  detected_patterns:     {prompt_analysis_result.detected_patterns}")
        
        print("\n⚡ STAGE 2: IN-GENERATION MONITORING")
        print("-" * 50)
        print(f"  entity_commitment_risk: {final_commitment_risk:.4f}")
        print(f"    committed_entities:   {len(commitment_result.committed_entities)}")
        print(f"    risk_signals:         {commitment_result.risk_signals[:3]}")
        print(f"  semantic_drift:         {final_drift:.4f}")
        print(f"    topic_shift:          {drift_result.topic_shift}")
        
        print("\n🔍 STAGE 3: POST-HOC VERIFICATION")
        print("-" * 50)
        print(f"  entity_novelty:         {final_entity_risk:.4f}")
        print(f"    novel_entities:       {final_entity_analysis.novel_entities[:5]}")
        print(f"  retrieval_risk:         {final_retrieval_risk:.4f}")
        print(f"    unverified_claims:    {final_retrieval_analysis.unverified_claims[:3]}")
        print(f"  self_consistency_risk:  {self_consistency_risk:.4f}")
        print(f"  semantic_entropy:       {semantic_entropy_risk:.4f}")
        print(f"  citation_verification:  {citation_verification_risk:.4f}")
        print("="*70)
        
        # Recompute final HRI with ALL signals
        final_hri = self._compute_running_hri(
            token_entropies,
            token_logprobs,
            # Stage 1
            prompt_risk=prompt_risk,
            entity_rarity=entity_rarity,
            cutoff_risk=cutoff_risk,
            compositional_novelty=compositional_novelty,
            # Stage 2
            entity_commitment_risk=final_commitment_risk,
            semantic_drift=final_drift,
            # Stage 3
            entity_risk=final_entity_risk,
            retrieval_risk=final_retrieval_risk,
            self_consistency_risk=self_consistency_risk,
            semantic_entropy=semantic_entropy_risk,
            citation_verification_risk=citation_verification_risk,
            debug=True,  # Enable debug output for final HRI
        )
        
        # Compute HRI slope and acceleration
        hri_slope, hri_acceleration = self._compute_hri_slope(hri_values)
        
        # Ensure final_hri is in valid range
        final_hri = max(0.0, min(1.0, final_hri))
        
        # Replace NaN/Inf in token_risks with 0.5
        token_risks = [r if not (math.isnan(r) or math.isinf(r)) else 0.5 for r in token_risks]
        # Replace NaN/Inf in hri_values with previous valid value or 0.2
        clean_hri_values = []
        last_valid = 0.2
        for h in hri_values:
            if math.isnan(h) or math.isinf(h):
                clean_hri_values.append(last_valid)
            else:
                clean_hri_values.append(h)
                last_valid = h
        hri_values = clean_hri_values
        
        clarifications = None
        if early_stopped or final_hri >= hri_threshold:
            clarifications = self._get_clarification_questions()
        
        return GenerationResult(
            text=full_text,
            prompt=prompt,
            tokens=token_strings,
            token_risks=token_risks,
            hri_values=hri_values,
            # Real-time intervention
            early_stopped=early_stopped,
            intervention_triggered=intervention_triggered,
            clarification_questions=clarifications,
            intervention_message=intervention_message,
            # Stage 1: Pre-hoc signals
            prompt_risk=prompt_risk,
            prompt_analysis=prompt_analysis_result,
            citation_risk=prompt_analysis_result.citation_risk,
            entity_rarity=entity_rarity,
            cutoff_risk=cutoff_risk,
            compositional_novelty=compositional_novelty,
            # Stage 2: In-generation signals
            entity_risk=final_entity_risk,
            entity_analysis=final_entity_analysis,
            novel_entities=final_entity_analysis.novel_entities,
            entity_commitment_risk=final_commitment_risk,
            commitment_result=commitment_result,
            semantic_drift=final_drift,
            drift_result=drift_result,
            # Stage 3: Post-hoc signals
            retrieval_risk=final_retrieval_risk,
            retrieval_analysis=final_retrieval_analysis,
            self_consistency_risk=self_consistency_risk,
            semantic_entropy=semantic_entropy_risk,
            citation_verification_risk=citation_verification_risk,
            posthoc_analysis=posthoc_analysis,
            # Full risk breakdown
            risk_breakdown=risk_breakdown,
            # Risk slope detection
            hri_slope=hri_slope,
            hri_acceleration=hri_acceleration,
            # Uncertainty metrics
            entropy_mean=entropy_mean,
            entropy_max=entropy_max,
            nll=nll,
            perplexity=ppl,
            logprob_std=std,
            logprob_variance=variance,
            input_ids=input_ids[0].detach().cpu(),
            generated_ids=current_ids[0].detach().cpu(),
            new_token_ids=torch.tensor(generated_tokens),
            vocab_size=self.vocab_size,
            # Legacy fields (not populated in new system)
            scores=None,
            topk_token_ids=None,
            topk_probs=None,
        )
