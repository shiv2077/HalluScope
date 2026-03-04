"""
HalluScope V4: Universal Hallucination Detection Layer

A purely mathematical hallucination detection system based on probability
geometry and internal representation analysis. No databases, no retrieval,
no external knowledge bases.

Core Innovation: Detecting hallucinations through model introspection using:
1. Semantic Energy (distribution confidence intensity)
2. Entropy and Logprob Variance (uncertainty patterns)
3. Confidence-Stability Gap (CSG) - catches confident hallucinations
4. Logit Lens Divergence (LLD) - reasoning stability across layers
5. Representation Collapse Detection (RCD) - hallucination loops

HRI Formula:
    HRI = sigmoid(
        w_entropy * entropy +
        w_energy * semantic_energy +
        w_variance * logprob_variance +
        w_csg * CSG +
        w_lld * LLD +
        w_rcd * representation_collapse +
        bias
    )

Author: HalluScope Team
Version: 4.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# HRI V4 Configuration - Pure Mathematical Signals
# ─────────────────────────────────────────────────────────────────────────────

HRI_WEIGHTS = {
    # Distribution Geometry Signals
    "w_entropy": 0.20,        # Token entropy (uncertainty)
    "w_energy": 0.15,         # Semantic energy (confidence intensity)
    "w_variance": 0.10,       # Logprob variance (instability)
    
    # Confidence-Stability Gap
    "w_csg": 0.25,            # CSG - catches confident hallucinations
    
    # Internal Representation Signals
    "w_lld": 0.15,            # Logit Lens Divergence (layer instability)
    "w_rcd": 0.15,            # Representation Collapse (loop detection)
    
    # Bias term (calibrated for sigmoid centering)
    "bias": -2.5,
}

# Thresholds
HRI_THRESHOLD = 0.65              # Early stopping threshold
HRI_INTERVENTION_THRESHOLD = 0.55  # Intervention/warning threshold
TOP_K_ENTROPY = 50                 # Top-k for fast entropy computation
SLIDING_WINDOW_SIZE = 10           # Window size for CSG stability computation
LLD_START_LAYER_FRACTION = 0.5     # Start LLD from 50% into layers

# Stop sequences - model-specific tokens that indicate end of generation
STOP_SEQUENCES = [
    "<|endoftext|>",
    "<|im_end|>",
    "</s>",
    "<|end|>",
    "Human:",
    "User:",
    "\nHuman:",
    "\nUser:",
    "\nAssistant:",
]

# Stability guards - prevent false positives on short generations
SEMANTIC_MIN_TOKENS = 8     # Minimum tokens before computing CSG
EARLY_STOP_MIN_TOKENS = 12  # Minimum tokens before allowing early stop
CSG_ARITHMETIC_WEIGHT = 0.2 # Reduced CSG weight for arithmetic prompts (default: 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes for Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V4Signals:
    """Per-token hallucination detection signals."""
    
    # Distribution Geometry
    entropy: float = 0.0              # Token entropy H = -Σ p log p
    semantic_energy: float = 0.0      # E = -logsumexp(logits)
    logprob: float = 0.0              # Log probability of selected token
    confidence: float = 0.0           # sigmoid(-mean_logprob)
    
    # Representation Analysis
    logit_lens_divergence: float = 0.0    # KL divergence across layers
    representation_collapse: float = 0.0   # cosine(h_t, h_{t-1})
    
    # Derived
    token_risk: float = 0.0           # Per-token HRI


@dataclass
class CSGResult:
    """Confidence-Stability Gap computation result."""
    
    confidence: float = 0.0           # sigmoid(-mean_logprob)
    semantic_stability: float = 0.0   # Mean cosine similarity to prompt
    csg: float = 0.0                  # confidence * (1 - stability)
    window_stabilities: List[float] = field(default_factory=list)


@dataclass
class LLDResult:
    """Logit Lens Divergence computation result."""
    
    lld: float = 0.0                  # Mean KL divergence across layers
    layer_divergences: List[float] = field(default_factory=list)
    num_layers_analyzed: int = 0


@dataclass 
class RCDResult:
    """Representation Collapse Detection result."""
    
    collapse_score: float = 0.0       # Mean cosine between consecutive hidden states
    per_token_collapse: List[float] = field(default_factory=list)
    is_collapsing: bool = False       # True if collapse_score > 0.95


@dataclass
class GenerationResult:
    """Complete generation result with V4 hallucination signals."""
    
    # Core output
    text: str = ""
    prompt: str = ""
    
    # Token-level data for visualization
    tokens: List[str] = field(default_factory=list)
    token_risks: List[float] = field(default_factory=list)
    
    # HRI trajectory (one value per generated token)
    hri_values: List[float] = field(default_factory=list)
    
    # Control flags
    early_stopped: bool = False
    intervention_triggered: bool = False
    intervention_message: Optional[str] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # V4 SIGNAL AGGREGATES
    # ─────────────────────────────────────────────────────────────────────────
    
    # Distribution Geometry (per-token lists)
    entropy_values: List[float] = field(default_factory=list)
    energy_values: List[float] = field(default_factory=list)
    logprob_values: List[float] = field(default_factory=list)
    
    # Aggregated metrics
    entropy_mean: float = 0.0
    entropy_max: float = 0.0
    semantic_energy_mean: float = 0.0
    logprob_variance: float = 0.0
    logprob_std: float = 0.0
    nll: float = 0.0                  # Mean NLL (for perplexity)
    perplexity: float = 1.0
    
    # Confidence-Stability Gap
    confidence: float = 0.0
    semantic_stability: float = 0.0
    confidence_stability_gap: float = 0.0
    csg_result: Optional[CSGResult] = None
    
    # Logit Lens Divergence
    logit_lens_divergence: float = 0.0
    lld_result: Optional[LLDResult] = None
    per_token_lld: List[float] = field(default_factory=list)
    
    # Representation Collapse
    representation_collapse: float = 0.0
    rcd_result: Optional[RCDResult] = None
    per_token_rcd: List[float] = field(default_factory=list)
    
    # Per-token CSG values
    per_token_csg: List[float] = field(default_factory=list)
    
    # Risk breakdown for UI/debugging
    risk_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # HRI dynamics
    hri_slope: float = 0.0
    hri_acceleration: float = 0.0
    
    # Model info
    vocab_size: int = 0
    
    # Token IDs (for advanced use)
    input_ids: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _is_arithmetic_prompt(prompt: str) -> bool:
    """
    Detect if prompt is a simple arithmetic question.
    
    Arithmetic prompts don't have meaningful semantic structure,
    so CSG-based detection should be reduced.
    
    Examples:
        "what is 2 + 2" -> True
        "52 + 48" -> True
        "calculate 100 / 5" -> True
        "explain quantum physics" -> False
    """
    import re
    
    prompt_lower = prompt.lower().strip()
    
    # Pattern 1: Pure arithmetic "52 + 48"
    if re.match(r'^\s*\d+\s*[\+\-\*\/\^]\s*\d+\s*[\?]?\s*$', prompt_lower):
        return True
    
    # Pattern 2: "what is X op Y" style
    if re.match(r'^(what\s+is|calculate|compute|solve|find)\s+\d+\s*[\+\-\*\/\^]\s*\d+', prompt_lower):
        return True
    
    # Pattern 3: Contains obvious arithmetic with numbers
    arithmetic_keywords = ['plus', 'minus', 'times', 'divided by', 'add', 'subtract', 'multiply', 'divide']
    has_arithmetic = any(kw in prompt_lower for kw in arithmetic_keywords)
    has_numbers = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', prompt_lower))
    
    if has_arithmetic or has_numbers:
        # Check it's short and simple
        word_count = len(prompt_lower.split())
        if word_count <= 10:
            return True
    
    return False


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    """Compute mean, handling empty lists and NaN values."""
    valid = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    return sum(valid) / len(valid) if valid else default


def _safe_variance(values: List[float], default: float = 0.0) -> float:
    """Compute variance, handling edge cases."""
    valid = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if len(valid) < 2:
        return default
    mean = sum(valid) / len(valid)
    return sum((v - mean) ** 2 for v in valid) / len(valid)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    
    return torch.dot(a_flat, b_flat).item() / (norm_a.item() * norm_b.item())


# ─────────────────────────────────────────────────────────────────────────────
# HalluScope V4 Backend
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend:
    """
    HalluScope V4: Universal Hallucination Detection Layer.
    
    Uses pure mathematical signals derived from probability distributions
    and internal representations to detect hallucinations without any
    external knowledge bases or retrieval systems.
    
    Key Features:
    - Semantic Energy: Measures confidence intensity via -logsumexp(logits)
    - Confidence-Stability Gap: Detects confident hallucinations
    - Logit Lens Divergence: Monitors reasoning stability across layers
    - Representation Collapse: Detects hallucination loops
    
    All signals are model-intrinsic and work across any domain.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        embedder_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize HalluScope V4 backend.
        
        Args:
            model_id: HuggingFace model ID for the generator.
            device: Device to use ("auto", "cuda", "cpu").
            embedder_model: Sentence transformer model for CSG computation.
        """
        self.model_id = model_id
        
        # Device detection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ HalluScope V4 running on: {gpu_name}")
            else:
                self.device = "cpu"
                print("⚠️  No GPU detected. Running on CPU (slower).")
        else:
            self.device = device
            print(f"HalluScope V4 running on: {device}")
        
        # Load tokenizer
        print(f"Loading tokenizer: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load generator model with hidden states enabled
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading model: {model_id} (dtype={dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            output_hidden_states=True,  # CRITICAL: Enable hidden states for LLD/RCD
        ).eval()
        print(f"✅ Generator model loaded on {self.model.device}")
        
        # Load sentence transformer for CSG (semantic stability)
        print(f"Loading embedding model: {embedder_model}...")
        self.embedder = SentenceTransformer(embedder_model, device=self.device)
        print(f"✅ Embedding model loaded")
        
        # Cache for prompt embedding (computed once per generation)
        self._prompt_embedding: Optional[torch.Tensor] = None
        
        # Previous hidden state for RCD computation
        self._prev_hidden_state: Optional[torch.Tensor] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 1: Semantic Energy
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_semantic_energy(self, logits: torch.Tensor) -> float:
        """
        Compute semantic energy from logits.
        
        E = -logsumexp(logits)
        
        Lower energy indicates stronger model confidence.
        Higher energy indicates uncertainty.
        
        Args:
            logits: Raw logits tensor [vocab_size] or [1, vocab_size]
            
        Returns:
            Semantic energy value (float).
        """
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        
        # E = -logsumexp(logits)
        energy = -torch.logsumexp(logits.float(), dim=-1).item()
        
        # Handle NaN/Inf
        if math.isnan(energy) or math.isinf(energy):
            return 0.0
        
        return energy
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 2: Entropy and Logprob Variance
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_token_entropy(
        self,
        probs: torch.Tensor,
        top_k: int = TOP_K_ENTROPY,
    ) -> float:
        """
        Compute token entropy from probability distribution.
        
        H = -Σ p * log(p)
        
        Uses top-k approximation for speed.
        
        Args:
            probs: Probability distribution [vocab_size]
            top_k: Number of top probabilities to use
            
        Returns:
            Entropy in nats.
        """
        try:
            top_probs, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            top_probs = torch.clamp(top_probs, min=1e-12, max=1.0)
            entropy = -(top_probs * torch.log(top_probs)).sum().item()
            
            if math.isnan(entropy) or math.isinf(entropy):
                return 3.0  # Default moderate entropy
            return entropy
        except Exception:
            return 3.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 3: Confidence-Stability Gap (CSG)
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_csg(
        self,
        prompt_text: str,
        generated_text: str,
        mean_logprob: float,
        window_size: int = SLIDING_WINDOW_SIZE,
    ) -> CSGResult:
        """
        Compute Confidence-Stability Gap.
        
        CSG = confidence * (1 - semantic_stability)
        
        Where:
        - confidence = sigmoid(-mean_logprob)
        - semantic_stability = mean cosine similarity between
          prompt embedding and generated text windows
        
        Key Insight:
        - Truthful answers: high confidence, high stability → LOW CSG
        - Uncertain answers: low confidence, medium stability → LOW CSG  
        - Confident hallucinations: high confidence, low stability → HIGH CSG
        
        Args:
            prompt_text: Original prompt text.
            generated_text: Generated text so far.
            mean_logprob: Mean log probability of generated tokens.
            window_size: Size of sliding window for stability computation.
            
        Returns:
            CSGResult with confidence, stability, and gap values.
        """
        # Compute confidence from logprobs
        # Higher probability → higher confidence
        confidence = _sigmoid(-mean_logprob)
        
        # Compute semantic stability using embeddings
        if not generated_text.strip():
            return CSGResult(
                confidence=confidence,
                semantic_stability=1.0,
                csg=0.0,
                window_stabilities=[],
            )
        
        # Encode prompt (use cached if available)
        if self._prompt_embedding is None:
            self._prompt_embedding = self.embedder.encode(
                prompt_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        
        # Encode generated text
        emb_generated = self.embedder.encode(
            generated_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        # Compute overall semantic similarity
        if self._prompt_embedding.device != emb_generated.device:
            emb_generated = emb_generated.to(self._prompt_embedding.device)
        
        stability = F.cosine_similarity(
            self._prompt_embedding.unsqueeze(0),
            emb_generated.unsqueeze(0),
            dim=1,
        ).item()
        
        # Clamp stability to [0, 1]
        stability = max(0.0, min(1.0, stability))
        
        # Compute CSG
        # High confidence + low stability = high risk
        csg = confidence * (1.0 - stability)
        
        return CSGResult(
            confidence=confidence,
            semantic_stability=stability,
            csg=csg,
            window_stabilities=[stability],
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 4: Logit Lens Divergence (LLD)
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_logit_lens_divergence(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> LLDResult:
        """
        Compute Logit Lens Divergence across transformer layers.
        
        Projects hidden states from each layer to vocabulary space
        using the unembedding matrix, then computes KL divergence
        between adjacent layer predictions.
        
        LLD = mean(KL(P_i || P_{i+1}))
        
        Where P_i = softmax(W_unembed @ h_i)
        
        Key Insight:
        - Stable reasoning: predictions converge as layers deepen → LOW LLD
        - Unstable reasoning: predictions fluctuate → HIGH LLD
        
        Args:
            hidden_states: Tuple of hidden states from all layers.
                          Each tensor has shape [batch, seq_len, hidden_dim].
        
        Returns:
            LLDResult with mean divergence and per-layer values.
        """
        divergences = []
        num_layers = len(hidden_states)
        
        # Only analyze later layers (where reasoning happens)
        start_layer = int(num_layers * LLD_START_LAYER_FRACTION)
        
        # Get unembedding matrix
        # This is typically the transpose of the embedding matrix
        try:
            if hasattr(self.model, 'lm_head'):
                unembed_weight = self.model.lm_head.weight  # [vocab_size, hidden_dim]
            elif hasattr(self.model, 'embed_out'):
                unembed_weight = self.model.embed_out.weight
            else:
                # Fallback: try to find the output embedding
                unembed_weight = None
                for name, module in self.model.named_modules():
                    if 'lm_head' in name or 'embed_out' in name:
                        if hasattr(module, 'weight'):
                            unembed_weight = module.weight
                            break
                
                if unembed_weight is None:
                    return LLDResult(lld=0.0, layer_divergences=[], num_layers_analyzed=0)
        except Exception:
            return LLDResult(lld=0.0, layer_divergences=[], num_layers_analyzed=0)
        
        # Move to CPU for large vocab to avoid OOM
        # The vocab is 150k+ tokens, so projection creates huge tensors
        unembed_weight_cpu = unembed_weight.cpu().float()
        
        prev_probs = None
        
        for i in range(start_layer, num_layers):
            try:
                # Get last token's hidden state for this layer (move to CPU)
                h_i = hidden_states[i][:, -1, :].cpu().float()  # [batch, hidden_dim]
                
                # Project to vocabulary space on CPU
                logits_i = torch.matmul(h_i, unembed_weight_cpu.T)
                probs_i = F.softmax(logits_i, dim=-1)
                
                if prev_probs is not None:
                    # Compute KL divergence: KL(prev || current)
                    # Add epsilon to avoid log(0)
                    kl = F.kl_div(
                        (probs_i + 1e-9).log(),
                        (prev_probs + 1e-9),
                        reduction='batchmean',
                    )
                    divergences.append(kl.item())
                
                prev_probs = probs_i
            except Exception:
                continue
        
        mean_lld = np.mean(divergences) if divergences else 0.0
        
        # Normalize LLD to [0, 1] range for HRI computation
        # Typical KL values range from 0 to ~10
        normalized_lld = min(1.0, mean_lld / 5.0)
        
        return LLDResult(
            lld=normalized_lld,
            layer_divergences=divergences,
            num_layers_analyzed=len(divergences),
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL 5: Representation Collapse Detection (RCD)
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_representation_collapse(
        self,
        current_hidden: torch.Tensor,
        collapse_threshold: float = 0.95,
    ) -> Tuple[float, bool]:
        """
        Detect representation collapse between consecutive tokens.
        
        collapse_score = cosine(h_t, h_{t-1})
        
        When similarity approaches 1.0 consistently, the model is
        stuck in a loop (hallucination mode).
        
        Args:
            current_hidden: Current token's hidden state [hidden_dim].
            collapse_threshold: Threshold for detecting collapse.
            
        Returns:
            Tuple of (collapse_score, is_collapsing).
        """
        if self._prev_hidden_state is None:
            self._prev_hidden_state = current_hidden.clone().detach()
            return 0.0, False
        
        # Compute cosine similarity
        collapse_score = _cosine_similarity(
            current_hidden,
            self._prev_hidden_state,
        )
        
        # Update previous hidden state
        self._prev_hidden_state = current_hidden.clone().detach()
        
        # Detect collapse
        is_collapsing = collapse_score > collapse_threshold
        
        # Normalize to [0, 1] for HRI
        # High similarity → high collapse risk
        normalized_collapse = max(0.0, (collapse_score - 0.5) * 2.0)
        
        return normalized_collapse, is_collapsing
    
    # ─────────────────────────────────────────────────────────────────────────
    # HRI Computation (V4 Formula)
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_hri(
        self,
        entropy: float,
        semantic_energy: float,
        logprob_variance: float,
        csg: float,
        lld: float,
        rcd: float,
        debug: bool = False,
    ) -> float:
        """
        Compute Hallucination Risk Index using V4 formula.
        
        HRI = sigmoid(
            w_entropy * entropy +
            w_energy * semantic_energy +
            w_variance * logprob_variance +
            w_csg * CSG +
            w_lld * LLD +
            w_rcd * RCD +
            bias
        )
        
        All inputs should be normalized to [0, 1] range.
        
        Args:
            entropy: Normalized token entropy.
            semantic_energy: Normalized semantic energy (higher = more uncertain).
            logprob_variance: Logprob variance (instability).
            csg: Confidence-Stability Gap.
            lld: Logit Lens Divergence.
            rcd: Representation Collapse Detection score.
            debug: If True, print detailed breakdown.
            
        Returns:
            HRI score in [0, 1].
        """
        # Compute weighted sum
        x = (
            HRI_WEIGHTS["w_entropy"] * entropy * 10 +
            HRI_WEIGHTS["w_energy"] * semantic_energy * 10 +
            HRI_WEIGHTS["w_variance"] * logprob_variance * 10 +
            HRI_WEIGHTS["w_csg"] * csg * 10 +
            HRI_WEIGHTS["w_lld"] * lld * 10 +
            HRI_WEIGHTS["w_rcd"] * rcd * 10 +
            HRI_WEIGHTS["bias"]
        )
        
        hri = _sigmoid(x)
        
        if debug:
            print("\n" + "=" * 70)
            print("=== HRI V4 SIGNAL BREAKDOWN ===")
            print("=" * 70)
            print("\n📊 DISTRIBUTION GEOMETRY")
            print("-" * 50)
            print(f"  entropy:          {entropy:.4f} → weighted: {HRI_WEIGHTS['w_entropy'] * entropy * 10:.4f}")
            print(f"  semantic_energy:  {semantic_energy:.4f} → weighted: {HRI_WEIGHTS['w_energy'] * semantic_energy * 10:.4f}")
            print(f"  logprob_variance: {logprob_variance:.4f} → weighted: {HRI_WEIGHTS['w_variance'] * logprob_variance * 10:.4f}")
            
            print("\n🎯 CONFIDENCE-STABILITY GAP")
            print("-" * 50)
            print(f"  CSG:              {csg:.4f} → weighted: {HRI_WEIGHTS['w_csg'] * csg * 10:.4f}")
            
            print("\n🔬 INTERNAL REPRESENTATION ANALYSIS")
            print("-" * 50)
            print(f"  LLD:              {lld:.4f} → weighted: {HRI_WEIGHTS['w_lld'] * lld * 10:.4f}")
            print(f"  RCD:              {rcd:.4f} → weighted: {HRI_WEIGHTS['w_rcd'] * rcd * 10:.4f}")
            
            print("\n📈 FINAL COMPUTATION")
            print("-" * 50)
            print(f"  bias:             {HRI_WEIGHTS['bias']:.4f}")
            print(f"  raw_score (x):    {x:.4f}")
            print(f"  HRI = σ(x):       {hri:.4f}")
            
            if hri >= 0.65:
                risk_label = "🔴 HIGH"
            elif hri >= 0.45:
                risk_label = "🟡 MEDIUM"
            else:
                risk_label = "🟢 LOW"
            print(f"  Risk Level:       {risk_label}")
            print("=" * 70 + "\n")
        
        return hri
    
    def _compute_token_risk(self, entropy: float, logprob: float) -> float:
        """
        Compute per-token risk score.
        
        Args:
            entropy: Token entropy.
            logprob: Log probability of token.
            
        Returns:
            Risk score in [0, 1].
        """
        if math.isnan(entropy) or math.isinf(entropy):
            entropy = 3.0
        if math.isnan(logprob) or math.isinf(logprob):
            logprob = -5.0
        
        scaled_entropy = entropy / 4.0
        scaled_nll = -logprob / 6.0
        
        x = 0.4 * scaled_entropy + 0.6 * scaled_nll - 1.0
        return _sigmoid(x)
    
    def _compute_hri_slope(
        self,
        hri_values: List[float],
        window: int = 5,
    ) -> Tuple[float, float]:
        """Compute HRI slope and acceleration."""
        if len(hri_values) < 3:
            return 0.0, 0.0
        
        recent = hri_values[-min(window, len(hri_values)):]
        slope = (recent[-1] - recent[0]) / len(recent)
        
        if len(hri_values) >= 6:
            earlier = hri_values[-6:-3]
            earlier_slope = (earlier[-1] - earlier[0]) / len(earlier) if len(earlier) > 1 else 0.0
            acceleration = slope - earlier_slope
        else:
            acceleration = 0.0
        
        return slope, acceleration
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Generation Method
    # ─────────────────────────────────────────────────────────────────────────
    
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
        debug: bool = False,
    ) -> GenerationResult:
        """
        Generate text with V4 hallucination detection.
        
        Uses manual autoregressive decoding to compute all signals
        at each token step:
        
        1. Semantic Energy (distribution confidence)
        2. Entropy (uncertainty)
        3. Logprob variance (instability)
        4. CSG (semantic drift from prompt)
        5. LLD (layer-wise prediction stability)
        6. RCD (representation collapse detection)
        
        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling constraint.
            do_sample: Whether to sample (default: True if temperature > 0).
            hri_threshold: HRI value that triggers early stopping.
            early_stop_enabled: Whether to enable early stopping.
            debug: If True, print detailed signal breakdowns.
        
        Returns:
            GenerationResult with all V4 signals and metrics.
        """
        if do_sample is None:
            do_sample = temperature > 0.0
        
        # Reset state
        self._prompt_embedding = None
        self._prev_hidden_state = None
        
        # Detect prompt type for signal weighting
        is_arithmetic = _is_arithmetic_prompt(prompt)
        csg_weight_multiplier = CSG_ARITHMETIC_WEIGHT if is_arithmetic else 1.0
        
        if debug and is_arithmetic:
            print(f"[DEBUG] Arithmetic prompt detected - reducing CSG weight to {csg_weight_multiplier}")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]
        
        # Storage
        generated_tokens: List[int] = []
        token_strings: List[str] = []
        
        # Per-token metrics
        entropy_values: List[float] = []
        energy_values: List[float] = []
        logprob_values: List[float] = []
        token_risks: List[float] = []
        hri_values: List[float] = []
        
        # V4 signals per token
        per_token_lld: List[float] = []
        per_token_rcd: List[float] = []
        per_token_csg: List[float] = []
        
        early_stopped = False
        intervention_triggered = False
        intervention_message = None
        current_ids = input_ids
        
        # ─────────────────────────────────────────────────────────────────────
        # Manual Autoregressive Decoding Loop
        # ─────────────────────────────────────────────────────────────────────
        
        for step in range(max_new_tokens):
            # Forward pass with hidden states
            outputs = self.model(
                input_ids=current_ids,
                output_hidden_states=True,
            )
            
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            hidden_states = outputs.hidden_states  # Tuple of tensors
            
            # Apply temperature
            if temperature > 0:
                scaled_logits = logits / temperature
            else:
                scaled_logits = logits
            
            # Compute probabilities
            probs = F.softmax(scaled_logits[0], dim=-1)
            
            # ───────────────────────────────────────────────────────────────
            # SIGNAL 1: Semantic Energy
            # ───────────────────────────────────────────────────────────────
            
            energy = self.compute_semantic_energy(logits[0])
            # Normalize: typical range is [-30, -15] for LLMs with large vocab
            # E = -logsumexp(logits) → more negative = more confident = lower risk
            # Map [-30, -15] → [0, 1] where 0 = confident, 1 = uncertain
            normalized_energy = max(0.0, min(1.0, (energy + 30.0) / 15.0))
            energy_values.append(normalized_energy)
            
            # ───────────────────────────────────────────────────────────────
            # SIGNAL 2: Entropy
            # ───────────────────────────────────────────────────────────────
            
            entropy = self.compute_token_entropy(probs)
            # Normalize: typical range is [0, 6], map to [0, 1]
            normalized_entropy = min(1.0, entropy / 6.0)
            entropy_values.append(normalized_entropy)
            
            # ───────────────────────────────────────────────────────────────
            # Sample next token
            # ───────────────────────────────────────────────────────────────
            
            if do_sample:
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    
                    sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum <= top_p
                    mask[0] = True
                    filtered_probs = sorted_probs * mask
                    filtered_probs = filtered_probs / filtered_probs.sum()
                    
                    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
                    local_idx = sorted_indices[sample_idx]
                    next_token = top_k_indices[local_idx].item()
                else:
                    next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(probs).item()
            
            # Get logprob of selected token
            logprob = torch.log(probs[next_token] + 1e-12).item()
            logprob_values.append(logprob)
            
            # Store token
            generated_tokens.append(next_token)
            token_str = self.tokenizer.decode([next_token])
            token_strings.append(token_str)
            
            # Per-token risk
            token_risk = self._compute_token_risk(entropy, logprob)
            token_risks.append(token_risk)
            
            # ───────────────────────────────────────────────────────────────
            # SIGNAL 4: Logit Lens Divergence (LLD)
            # ───────────────────────────────────────────────────────────────
            
            lld_result = self.compute_logit_lens_divergence(hidden_states)
            
            per_token_lld.append(lld_result.lld)
            
            # ───────────────────────────────────────────────────────────────
            # SIGNAL 5: Representation Collapse Detection (RCD)
            # ───────────────────────────────────────────────────────────────
            
            # Use last layer's last token hidden state
            current_hidden = hidden_states[-1][:, -1, :].squeeze(0)
            rcd_score, is_collapsing = self.compute_representation_collapse(current_hidden)
            per_token_rcd.append(rcd_score)
            
            # ───────────────────────────────────────────────────────────────
            # SIGNAL 3: CSG (computed after minimum tokens for stability)
            # ───────────────────────────────────────────────────────────────
            
            # Only compute CSG after enough tokens for stable embeddings
            if len(logprob_values) >= SEMANTIC_MIN_TOKENS:
                mean_logprob = _safe_mean(logprob_values)
                generated_text = "".join(token_strings)
                csg_result = self.compute_csg(prompt, generated_text, mean_logprob)
                # Apply arithmetic weight reduction
                adjusted_csg = csg_result.csg * csg_weight_multiplier
                per_token_csg.append(adjusted_csg)
            elif len(logprob_values) >= 3:
                # Minimal CSG for early tokens (heavily dampened)
                mean_logprob = _safe_mean(logprob_values)
                generated_text = "".join(token_strings)
                csg_result = self.compute_csg(prompt, generated_text, mean_logprob)
                # Heavily dampen early CSG to prevent false positives
                dampened_csg = csg_result.csg * 0.1 * csg_weight_multiplier
                per_token_csg.append(dampened_csg)
            else:
                per_token_csg.append(0.0)
            
            # ───────────────────────────────────────────────────────────────
            # Compute Running HRI
            # ───────────────────────────────────────────────────────────────
            
            # Aggregate metrics so far
            mean_entropy = _safe_mean(entropy_values)
            mean_energy = _safe_mean(energy_values)
            logprob_var = _safe_variance(logprob_values)
            normalized_var = min(1.0, logprob_var / 4.0)  # Normalize
            
            mean_lld = _safe_mean(per_token_lld)
            mean_rcd = _safe_mean(per_token_rcd)
            current_csg = per_token_csg[-1] if per_token_csg else 0.0
            
            # Compute HRI
            current_hri = self.compute_hri(
                entropy=mean_entropy,
                semantic_energy=mean_energy,
                logprob_variance=normalized_var,
                csg=current_csg,
                lld=mean_lld,
                rcd=mean_rcd,
                debug=debug and step == 0,  # Only debug first token
            )
            hri_values.append(current_hri)
            
            # ───────────────────────────────────────────────────────────────
            # Check for intervention/early stopping
            # ───────────────────────────────────────────────────────────────
            
            if current_hri >= HRI_INTERVENTION_THRESHOLD and not intervention_triggered:
                # Only trigger intervention after enough tokens
                if step >= SEMANTIC_MIN_TOKENS:
                    intervention_triggered = True
                    intervention_message = (
                        f"⚠️ High hallucination risk detected (HRI: {current_hri:.2f})\n\n"
                        f"The model shows signs of:\n"
                    )
                    if current_csg > 0.3:
                        intervention_message += "• High confidence with semantic drift from prompt\n"
                    if mean_lld > 0.3:
                        intervention_message += "• Unstable reasoning across internal layers\n"
                    if mean_rcd > 0.3:
                        intervention_message += "• Representation collapse (repetition risk)\n"
                    intervention_message += "\nPlease verify the output carefully."
            
            # Early stopping - only after minimum tokens to prevent false alarms
            if early_stop_enabled and step >= EARLY_STOP_MIN_TOKENS and current_hri >= hri_threshold:
                early_stopped = True
                if debug:
                    print(f"⚠️ Early stopping at step {step}: HRI={current_hri:.4f}")
                break
            
            # Update input IDs for next iteration
            next_token_tensor = torch.tensor([[next_token]], device=self.model.device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
            
            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Check for stop sequences in generated text
            current_text = "".join(token_strings)
            stop_found = False
            for stop_seq in STOP_SEQUENCES:
                if stop_seq in current_text:
                    # Trim at stop sequence
                    stop_idx = current_text.find(stop_seq)
                    if stop_idx > 0:
                        # Recalculate token_strings to exclude content after stop
                        trimmed_text = current_text[:stop_idx]
                        # Find how many tokens to keep
                        test_len = 0
                        keep_tokens = 0
                        for i, ts in enumerate(token_strings):
                            test_len += len(ts)
                            if test_len > stop_idx:
                                keep_tokens = i
                                break
                            keep_tokens = i + 1
                        # Trim all lists to match
                        if keep_tokens > 0 and keep_tokens < len(token_strings):
                            token_strings = token_strings[:keep_tokens]
                            generated_tokens = generated_tokens[:keep_tokens]
                            entropy_values = entropy_values[:keep_tokens]
                            energy_values = energy_values[:keep_tokens]
                            logprob_values = logprob_values[:keep_tokens]
                            token_risks = token_risks[:keep_tokens]
                            hri_values = hri_values[:keep_tokens]
                            per_token_lld = per_token_lld[:keep_tokens]
                            per_token_rcd = per_token_rcd[:keep_tokens]
                            per_token_csg = per_token_csg[:keep_tokens]
                    stop_found = True
                    break
            
            if stop_found:
                break
        
        # ─────────────────────────────────────────────────────────────────────
        # Post-generation aggregation
        # ─────────────────────────────────────────────────────────────────────
        
        generated_text = "".join(token_strings)
        
        # Final cleanup: remove any remaining stop sequences from output
        for stop_seq in STOP_SEQUENCES:
            if stop_seq in generated_text:
                generated_text = generated_text.split(stop_seq)[0]
        
        # Strip trailing whitespace/newlines
        generated_text = generated_text.rstrip()
        
        # Final CSG computation with full generated text
        if logprob_values:
            final_csg = self.compute_csg(
                prompt,
                generated_text,
                _safe_mean(logprob_values),
            )
        else:
            final_csg = CSGResult()
        
        # Aggregate metrics
        entropy_mean = _safe_mean(entropy_values)
        entropy_max = max(entropy_values) if entropy_values else 0.0
        semantic_energy_mean = _safe_mean(energy_values)
        logprob_var = _safe_variance(logprob_values)
        logprob_std = math.sqrt(logprob_var) if logprob_var > 0 else 0.0
        nll = -_safe_mean(logprob_values) if logprob_values else 0.0
        perplexity = math.exp(nll) if nll < 20 else float('inf')
        
        # Final LLD/RCD aggregates
        final_lld = _safe_mean(per_token_lld)
        final_rcd = _safe_mean(per_token_rcd)
        
        # HRI dynamics
        hri_slope, hri_accel = self._compute_hri_slope(hri_values)
        
        # Risk breakdown
        risk_breakdown = {
            "entropy": entropy_mean,
            "semantic_energy": semantic_energy_mean,
            "logprob_variance": min(1.0, logprob_var / 4.0),
            "csg": final_csg.csg,
            "lld": final_lld,
            "rcd": final_rcd,
            "confidence": final_csg.confidence,
            "semantic_stability": final_csg.semantic_stability,
        }
        
        return GenerationResult(
            text=prompt + generated_text,
            prompt=prompt,
            tokens=token_strings,
            token_risks=token_risks,
            hri_values=hri_values,
            early_stopped=early_stopped,
            intervention_triggered=intervention_triggered,
            intervention_message=intervention_message,
            
            # Per-token V4 signals
            entropy_values=entropy_values,
            energy_values=energy_values,
            logprob_values=logprob_values,
            
            # Aggregates
            entropy_mean=entropy_mean,
            entropy_max=entropy_max,
            semantic_energy_mean=semantic_energy_mean,
            logprob_variance=logprob_var,
            logprob_std=logprob_std,
            nll=nll,
            perplexity=perplexity,
            
            # CSG
            confidence=final_csg.confidence,
            semantic_stability=final_csg.semantic_stability,
            confidence_stability_gap=final_csg.csg,
            csg_result=final_csg,
            
            # LLD
            logit_lens_divergence=final_lld,
            lld_result=LLDResult(
                lld=final_lld,
                layer_divergences=[],
                num_layers_analyzed=0,
            ),
            per_token_lld=per_token_lld,
            
            # RCD
            representation_collapse=final_rcd,
            rcd_result=RCDResult(
                collapse_score=final_rcd,
                per_token_collapse=per_token_rcd,
                is_collapsing=final_rcd > 0.8,
            ),
            per_token_rcd=per_token_rcd,
            
            # Per-token CSG
            per_token_csg=per_token_csg,
            
            # Risk breakdown
            risk_breakdown=risk_breakdown,
            hri_slope=hri_slope,
            hri_acceleration=hri_accel,
            
            vocab_size=self.vocab_size,
            input_ids=input_ids,
            generated_ids=current_ids,
        )
    
    def analyze_generation(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method matching the HalluScopeV4 interface.
        
        Returns a simplified dict with key signals.
        """
        result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            debug=debug,
        )
        
        final_hri = result.hri_values[-1] if result.hri_values else 0.0
        
        return {
            "hri": final_hri,
            "generated_text": result.text[len(prompt):],
            "signals": {
                "entropy": result.entropy_mean,
                "semantic_energy": result.semantic_energy_mean,
                "confidence": result.confidence,
                "semantic_stability": result.semantic_stability,
                "csg": result.confidence_stability_gap,
                "logit_lens_divergence": result.logit_lens_divergence,
                "representation_collapse": result.representation_collapse,
                "logprob_variance": result.logprob_variance,
            },
            "per_token": {
                "hri": result.hri_values,
                "csg": result.per_token_csg,
                "lld": result.per_token_lld,
                "rcd": result.per_token_rcd,
            },
            "early_stopped": result.early_stopped,
            "intervention_message": result.intervention_message,
        }
