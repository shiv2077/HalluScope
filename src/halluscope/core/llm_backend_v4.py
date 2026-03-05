"""
HalluScope V4.2: Universal Hallucination Detection Layer

V4.2 adds semantic fabrication detection signals that catch confident
hallucinations — the failure mode where the model invents plausible-
sounding content with low entropy and high stability.

Core Signals (V4.1 — token uncertainty):
1. Semantic Energy (E = -logsumexp)
2. Entropy / Logprob Variance
3. Confidence-Stability Gap (CSG)
4. Logit Lens Divergence (LLD)
5. Representation Collapse Detection (RCD)

Semantic Fabrication Signals (V4.2 — claim verification):
6. Semantic Novelty — drift from prompt topic
7. Entity Expansion — novel content word ratio
8. Claim Density — assertions without evidence

HRI = sigmoid(Σ w_i * signal_i * 10 + bias)

Version: 4.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class Regime:
    SHORT_ANSWER = "short_answer"    # prompt_tokens <= 6
    FACTUAL = "factual"              # prompt_tokens <= 20
    GENERATIVE = "generative"        # prompt_tokens > 20


# Stage 2 weights — token-level signals ONLY (no novelty/entity/claim)
WEIGHT_PROFILES = {
    Regime.SHORT_ANSWER: {
        "w_entropy": 0.40, "w_energy": 0.00, "w_variance": 0.25,
        "w_csg": 0.15, "w_lld": 0.10, "w_rcd": 0.10,
        "bias": -2.5,
    },
    Regime.FACTUAL: {
        "w_entropy": 0.20, "w_energy": 0.05, "w_variance": 0.15,
        "w_csg": 0.20, "w_lld": 0.20, "w_rcd": 0.20,
        "bias": -2.0,
    },
    Regime.GENERATIVE: {
        "w_entropy": 0.15, "w_energy": 0.10, "w_variance": 0.15,
        "w_csg": 0.20, "w_lld": 0.20, "w_rcd": 0.20,
        "bias": -2.0,
    },
}

HRI_WEIGHTS = WEIGHT_PROFILES[Regime.GENERATIVE]

# Token gating thresholds
ENTROPY_MIN_TOKENS = 2
SEMANTIC_MIN_TOKENS = 8
LLD_MIN_TOKENS = 10
RCD_MIN_TOKENS = 10
EARLY_STOP_MIN_TOKENS = 12

# Signal clamping
CSG_MAX = 0.6
LLD_MAX = 0.5
RCD_MAX = 0.5
NOVELTY_MAX = 1.0
ENTITY_MAX = 1.0
CLAIM_MAX = 1.0

# Performance: expensive signal intervals
CSG_COMPUTE_INTERVAL = 16
LLD_COMPUTE_INTERVAL = 8
LLD_SAMPLE_LAYERS = 4

# Thresholds
HRI_THRESHOLD = 0.65
HRI_INTERVENTION_THRESHOLD = 0.55
TOP_K_ENTROPY = 50
SLIDING_WINDOW_SIZE = 8
LLD_START_LAYER_FRACTION = 0.5

# Stop sequences
STOP_SEQUENCES = [
    "<|endoftext|>", "<|im_end|>", "</s>", "<|end|>",
    "Human:", "User:", "\nHuman:", "\nUser:", "\nAssistant:",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V4Signals:
    entropy: float = 0.0
    semantic_energy: float = 0.0
    logprob: float = 0.0
    confidence: float = 0.0
    logit_lens_divergence: float = 0.0
    representation_collapse: float = 0.0
    token_risk: float = 0.0


@dataclass
class CSGResult:
    confidence: float = 0.0
    semantic_stability: float = 0.0
    csg: float = 0.0
    window_stabilities: List[float] = field(default_factory=list)


@dataclass
class LLDResult:
    lld: float = 0.0
    layer_divergences: List[float] = field(default_factory=list)
    num_layers_analyzed: int = 0


@dataclass
class RCDResult:
    collapse_score: float = 0.0
    per_token_collapse: List[float] = field(default_factory=list)
    is_collapsing: bool = False


@dataclass
class GenerationResult:
    text: str = ""
    prompt: str = ""
    tokens: List[str] = field(default_factory=list)
    token_risks: List[float] = field(default_factory=list)
    hri_values: List[float] = field(default_factory=list)
    early_stopped: bool = False
    intervention_triggered: bool = False
    intervention_message: Optional[str] = None

    entropy_values: List[float] = field(default_factory=list)
    energy_values: List[float] = field(default_factory=list)
    logprob_values: List[float] = field(default_factory=list)

    entropy_mean: float = 0.0
    entropy_max: float = 0.0
    semantic_energy_mean: float = 0.0
    logprob_variance: float = 0.0
    logprob_std: float = 0.0
    nll: float = 0.0
    perplexity: float = 1.0

    confidence: float = 0.0
    semantic_stability: float = 0.0
    confidence_stability_gap: float = 0.0
    csg_result: Optional[CSGResult] = None

    logit_lens_divergence: float = 0.0
    lld_result: Optional[LLDResult] = None
    per_token_lld: List[float] = field(default_factory=list)

    representation_collapse: float = 0.0
    rcd_result: Optional[RCDResult] = None
    per_token_rcd: List[float] = field(default_factory=list)

    per_token_csg: List[float] = field(default_factory=list)

    # V4.2 Semantic fabrication signals (computed post-hoc in V5)
    novelty: float = 0.0
    entity_expansion: float = 0.0
    claim_density: float = 0.0
    per_token_novelty: List[float] = field(default_factory=list)

    # V5: Prompt perplexity (from prefill — Stage 1)
    prompt_perplexity: float = 1.0
    prompt_logprobs: List[float] = field(default_factory=list)
    min_prompt_logprob: float = 0.0
    low_prob_fraction: float = 0.0

    risk_breakdown: Dict[str, float] = field(default_factory=dict)
    hri_slope: float = 0.0
    hri_acceleration: float = 0.0
    vocab_size: int = 0
    input_ids: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _detect_regime(prompt: str, tokenizer) -> str:
    n = len(tokenizer.encode(prompt, add_special_tokens=False))
    if n <= 6:
        return Regime.SHORT_ANSWER
    elif n <= 20:
        return Regime.FACTUAL
    return Regime.GENERATIVE


def _compute_hri_slope(hri_values: List[float], window: int = 4) -> float:
    if len(hri_values) < window:
        return 0.0
    recent = hri_values[-window:]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    return sum(diffs) / len(diffs) if diffs else 0.0


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    valid = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    return sum(valid) / len(valid) if valid else default


def _safe_variance(values: List[float], default: float = 0.0) -> float:
    valid = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if len(valid) < 2:
        return default
    mean = sum(valid) / len(valid)
    return sum((v - mean) ** 2 for v in valid) / len(valid)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    na = torch.norm(a_f)
    nb = torch.norm(b_f)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return torch.dot(a_f, b_f).item() / (na.item() * nb.item())


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
]

_EVIDENCE_PATTERNS = [
    "according to", "published in", "cited by", "referenced in",
    "doi:", "isbn:", "arxiv:", "et al.", "peer-reviewed",
    "journal", "proceedings", "volume", "experiment", "data shows",
    "evidence suggests", "empirical", "measured", "observed",
    "i don't know", "i'm not sure", "cannot confirm", "it is unclear",
    "it is uncertain", "however", "although", "on the other hand",
    "note that", "disclaimer", "not verified",
]


def _compute_entity_expansion(prompt: str, gen_text: str) -> float:
    """Fraction of answer content words NOT found in the prompt.

    High values indicate the model is introducing novel vocabulary,
    which is a signal of semantic fabrication when combined with
    low uncertainty.
    """
    prompt_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', prompt))
    prompt_words -= _STOP_WORDS
    gen_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', gen_text))
    gen_content = gen_words - _STOP_WORDS
    if not gen_content:
        return 0.0
    new_content = gen_content - prompt_words
    return min(1.0, len(new_content) / max(len(gen_content), 1))


def _compute_claim_density(gen_text: str) -> float:
    """Ratio of assertive language to evidence/hedging markers.

    High values indicate the model is making many ungrounded claims
    without citing evidence, expressing uncertainty, or hedging.
    A confident hallucination will have high claim density and low
    entropy — the exact pattern our detector must catch.
    """
    text_lower = gen_text.lower()
    assertions = sum(1 for p in _ASSERTION_PATTERNS if p in text_lower)
    evidence = sum(1 for p in _EVIDENCE_PATTERNS if p in text_lower)
    if assertions + evidence == 0:
        return 0.0
    return min(1.0, assertions / (assertions + evidence + 1))


# ─────────────────────────────────────────────────────────────────────────────
# LLM Backend
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend:
    """
    HalluScope V5 — Three-Stage KV-cached generation.

    Stage 1: Prompt perplexity extracted from prefill (free).
    Stage 2: Per-token HRI with entropy, energy, CSG, LLD, RCD.
    Stage 3: Post-hoc novelty, claim density, confidence paradox.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        embedder_model: str = "all-MiniLM-L6-v2",
    ):
        self.model_id = model_id

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"✅ HalluScope V5 on: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("⚠️  No GPU — running on CPU.")
        else:
            self.device = device

        # Tokenizer
        print(f"Loading tokenizer: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        # Model (4-bit quantized on GPU for speed, fp32 on CPU)
        if self.device == "cuda":
            # Clear GPU memory before loading to maximize available space
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print(f"Loading model: {model_id} (4-bit quantized, compute fp16)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
                max_memory={0: "5GiB", "cpu": "16GiB"},
                low_cpu_mem_usage=True,
            ).eval()
        else:
            print(f"Loading model: {model_id} (fp32 CPU)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            ).eval()
        print(f"✅ Model loaded on {self.model.device}")

        # Pre-cache unembed matrix on GPU for LLD
        self._unembed_weight: Optional[torch.Tensor] = None
        self._init_unembed_weight()

        # Pre-compute sparse LLD layer indices
        num_layers = self.model.config.num_hidden_layers
        start = int(num_layers * LLD_START_LAYER_FRACTION)
        total = num_layers - start
        if total <= LLD_SAMPLE_LAYERS:
            self._lld_layers = list(range(start, num_layers))
        else:
            step = total / LLD_SAMPLE_LAYERS
            self._lld_layers = [start + int(i * step) for i in range(LLD_SAMPLE_LAYERS)]
        print(f"  LLD layers: {self._lld_layers} / {num_layers}")

        # Sentence transformer for CSG
        print(f"Loading embedder: {embedder_model}...")
        self.embedder = SentenceTransformer(embedder_model, device=self.device)
        print("✅ Embedder loaded")

        # Per-generation state
        self._prompt_embedding: Optional[torch.Tensor] = None
        self._prev_hidden: Optional[torch.Tensor] = None

    def _init_unembed_weight(self):
        try:
            if hasattr(self.model, "lm_head"):
                self._unembed_weight = self.model.lm_head.weight.detach()
            elif hasattr(self.model, "embed_out"):
                self._unembed_weight = self.model.embed_out.weight.detach()
            else:
                for name, mod in self.model.named_modules():
                    if "lm_head" in name or "embed_out" in name:
                        if hasattr(mod, "weight"):
                            self._unembed_weight = mod.weight.detach()
                            break
        except Exception:
            self._unembed_weight = None
        if self._unembed_weight is not None:
            print(f"  Unembed matrix: {self._unembed_weight.shape} on {self._unembed_weight.device}")

    # ── Signals ──────────────────────────────────────────────────────────────

    def compute_semantic_energy(self, logits: torch.Tensor) -> float:
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        energy = -torch.logsumexp(logits.float(), dim=-1).item()
        return 0.0 if (math.isnan(energy) or math.isinf(energy)) else energy

    def compute_token_entropy(self, probs: torch.Tensor, top_k: int = TOP_K_ENTROPY) -> float:
        try:
            top_p, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            top_p = torch.clamp(top_p, min=1e-12, max=1.0)
            h = -(top_p * torch.log(top_p)).sum().item()
            return 3.0 if (math.isnan(h) or math.isinf(h)) else h
        except Exception:
            return 3.0

    def compute_csg(self, prompt: str, gen_text: str, mean_logprob: float) -> CSGResult:
        conf = _sigmoid(-mean_logprob)
        if not gen_text.strip():
            return CSGResult(confidence=conf, semantic_stability=1.0, csg=0.0)
        if self._prompt_embedding is None:
            self._prompt_embedding = self.embedder.encode(
                prompt, convert_to_tensor=True, normalize_embeddings=True,
            )
        emb_gen = self.embedder.encode(
            gen_text, convert_to_tensor=True, normalize_embeddings=True,
        )
        if self._prompt_embedding.device != emb_gen.device:
            emb_gen = emb_gen.to(self._prompt_embedding.device)
        stab = F.cosine_similarity(
            self._prompt_embedding.unsqueeze(0), emb_gen.unsqueeze(0), dim=1
        ).item()
        stab = max(0.0, min(1.0, stab))
        csg = conf * (1.0 - stab)
        return CSGResult(confidence=conf, semantic_stability=stab, csg=csg, window_stabilities=[stab])

    def compute_lld(self, hidden_states: Tuple[torch.Tensor, ...]) -> LLDResult:
        if self._unembed_weight is None:
            return LLDResult()
        divergences = []
        prev_probs = None
        unembed = self._unembed_weight
        for idx in self._lld_layers:
            if idx >= len(hidden_states):
                continue
            try:
                h = hidden_states[idx][:, -1, :]
                logits_i = torch.matmul(h.float(), unembed.float().T)
                probs_i = F.softmax(logits_i, dim=-1)
                if prev_probs is not None:
                    kl = F.kl_div((probs_i + 1e-9).log(), prev_probs + 1e-9, reduction="batchmean")
                    divergences.append(kl.item())
                prev_probs = probs_i
            except Exception:
                continue
        mean_lld = float(np.mean(divergences)) if divergences else 0.0
        return LLDResult(lld=min(1.0, mean_lld / 5.0), layer_divergences=divergences,
                         num_layers_analyzed=len(divergences))

    def compute_rcd(self, current_hidden: torch.Tensor) -> Tuple[float, bool]:
        if self._prev_hidden is None:
            self._prev_hidden = current_hidden.clone().detach()
            return 0.0, False
        score = _cosine_similarity(current_hidden, self._prev_hidden)
        self._prev_hidden = current_hidden.clone().detach()
        norm = max(0.0, (score - 0.5) * 2.0)
        return norm, score > 0.95

    # ── HRI ──────────────────────────────────────────────────────────────────

    def compute_hri_dynamic(
        self, entropy: float, semantic_energy: float, logprob_variance: float,
        csg: float, lld: float, rcd: float,
        weights: Dict[str, float], prompt_risk_bias: float = 0.0,
        debug: bool = False,
    ) -> float:
        """Stage 2 HRI: token-level signals only + pre-hoc bias."""
        x = (
            weights["w_entropy"] * entropy * 10
            + weights["w_energy"] * semantic_energy * 10
            + weights["w_variance"] * logprob_variance * 10
            + weights["w_csg"] * csg * 10
            + weights["w_lld"] * lld * 10
            + weights["w_rcd"] * rcd * 10
            + weights["bias"]
            + prompt_risk_bias
        )
        hri = _sigmoid(x)
        if debug:
            print(f"  [HRI] x={x:.4f} → {hri:.4f} (bias_prehoc={prompt_risk_bias:.3f})")
        return hri

    def _token_risk(self, entropy: float, logprob: float) -> float:
        e = 3.0 if (math.isnan(entropy) or math.isinf(entropy)) else entropy
        lp = -5.0 if (math.isnan(logprob) or math.isinf(logprob)) else logprob
        return _sigmoid(0.4 * (e / 4.0) + 0.6 * (-lp / 6.0) - 1.0)

    def _hri_slope_accel(self, hri_values: List[float], window: int = 5) -> Tuple[float, float]:
        if len(hri_values) < 3:
            return 0.0, 0.0
        recent = hri_values[-min(window, len(hri_values)):]
        slope = (recent[-1] - recent[0]) / len(recent)
        if len(hri_values) >= 6:
            earlier = hri_values[-6:-3]
            es = (earlier[-1] - earlier[0]) / len(earlier) if len(earlier) > 1 else 0.0
            accel = slope - es
        else:
            accel = 0.0
        return slope, accel

    # ── Main Generation (KV-cached) ─────────────────────────────────────────

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
        prompt_risk_bias: float = 0.0,
        debug: bool = False,
    ) -> GenerationResult:
        """
        V5 KV-cached generation with three-stage signal extraction.

        Stage 1: Prompt perplexity extracted from prefill (FREE).
        Stage 2: Per-token HRI using entropy, energy, variance, CSG, LLD, RCD.
        Stage 3: Post-hoc signals computed on full text after generation.
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        # Reset
        self._prompt_embedding = None
        self._prev_hidden = None

        # Clear fragmented GPU memory before generation
        if self.device == "cuda":
            torch.cuda.empty_cache()

        regime = _detect_regime(prompt, self.tokenizer)
        weights = WEIGHT_PROFILES[regime]

        if debug:
            pt = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            print(f"\n{'='*60}\nHALLUSCOPE V5 | regime={regime} (pt={pt}) | 3-STAGE\n{'='*60}")

        # ── Tokenize with chat template (Instruct models need ChatML) ──
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            chat_result = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
            # apply_chat_template may return BatchEncoding or plain list
            if hasattr(chat_result, "keys"):
                input_ids_list = chat_result["input_ids"]
            else:
                input_ids_list = chat_result
            input_ids = torch.tensor([input_ids_list], device=self.model.device)
        except Exception:
            # Fallback: raw tokenization if chat template unavailable
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            input_ids_list = input_ids[0].tolist()

        # Track where user prompt tokens start for perplexity measurement
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        n_user_tokens = len(raw_prompt_ids)
        # In ChatML: ...user\n<prompt><|im_end|>\n<|im_start|>assistant\n
        # User tokens end at: total - 4 (im_end, newline, im_start_assistant, newline)
        # User tokens start at: total - 4 - n_user_tokens
        n_template = len(input_ids_list)
        user_token_end = n_template - 4  # skip trailing assistant prompt tokens
        user_token_start = max(0, user_token_end - n_user_tokens)

        # ── Prefill (full prompt, get KV cache) ──
        prefill = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        past_kv = prefill.past_key_values
        self._prev_hidden = prefill.hidden_states[-1][:, -1, :].squeeze(0).clone().detach()

        # ── Stage 1: Extract prompt perplexity from prefill (FREE) ──
        # Only measure USER prompt tokens, not chat template tokens
        prompt_logprobs_list: List[float] = []
        prompt_ppl = 1.0
        min_prompt_lp = 0.0
        low_prob_frac = 0.0
        if input_ids.size(1) > 1:
            prompt_log_probs = F.log_softmax(prefill.logits[0, :-1], dim=-1)
            prompt_token_ids = input_ids[0, 1:]
            all_token_lps = prompt_log_probs.gather(
                1, prompt_token_ids.unsqueeze(1)
            ).squeeze(1)
            # Extract only user-prompt token logprobs (skip template overhead)
            if n_user_tokens > 0 and user_token_start < len(all_token_lps):
                user_end = min(user_token_start + n_user_tokens, len(all_token_lps))
                user_lps = all_token_lps[user_token_start:user_end]
                prompt_logprobs_list = user_lps.tolist()
            else:
                prompt_logprobs_list = all_token_lps.tolist()
            prompt_nll_val = -sum(prompt_logprobs_list) / max(len(prompt_logprobs_list), 1)
            prompt_ppl = math.exp(min(prompt_nll_val, 20))
            # Use 25th percentile instead of min — robust against template outliers
            sorted_lps = sorted(prompt_logprobs_list)
            p25_idx = max(0, len(sorted_lps) // 4)
            min_prompt_lp = sorted_lps[p25_idx] if sorted_lps else 0.0
            low_prob_frac = sum(
                1 for lp in prompt_logprobs_list if lp < -5.0
            ) / max(len(prompt_logprobs_list), 1)
            if debug:
                print(f"  [Stage1] prompt_ppl={prompt_ppl:.2f} min_lp={min_prompt_lp:.3f} "
                      f"low_frac={low_prob_frac:.3f} (user_tokens={n_user_tokens})")

        # Storage
        tokens_str: List[str] = []
        tokens_id: List[int] = []
        entropy_vals: List[float] = []
        energy_vals: List[float] = []
        logprob_vals: List[float] = []
        risk_vals: List[float] = []
        hri_vals: List[float] = []
        pt_lld: List[float] = []
        pt_rcd: List[float] = []
        pt_csg: List[float] = []

        cached_csg = 0.0
        cached_lld = 0.0
        cached_rcd = 0.0
        early_stopped = False
        intervention_triggered = False
        intervention_msg = None

        logits = prefill.logits[:, -1, :]  # [1, vocab]

        for step in range(max_new_tokens):
            # Temperature
            sc_logits = logits / temperature if temperature > 0 else logits
            probs = F.softmax(sc_logits[0], dim=-1)

            # Signal 1: Energy
            energy = self.compute_semantic_energy(logits[0])
            norm_energy = max(0.0, min(1.0, (energy + 30.0) / 15.0))
            energy_vals.append(norm_energy)

            # Signal 2: Entropy
            ent = self.compute_token_entropy(probs)
            norm_ent = min(1.0, ent / 6.0)
            entropy_vals.append(norm_ent)

            # ── Sample ──
            if do_sample:
                if top_k > 0:
                    tk_p, tk_i = torch.topk(probs, top_k)
                    tk_p = tk_p / tk_p.sum()
                    sp, si = torch.sort(tk_p, descending=True)
                    cm = torch.cumsum(sp, dim=-1)
                    mask = cm <= top_p
                    mask[0] = True
                    fp = sp * mask
                    fp = fp / fp.sum()
                    sx = torch.multinomial(fp, 1)
                    li = si[sx]
                    next_tok = tk_i[li].item()
                else:
                    next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = torch.argmax(probs).item()

            lp = torch.log(probs[next_tok] + 1e-12).item()
            logprob_vals.append(lp)

            tokens_id.append(next_tok)
            ts = self.tokenizer.decode([next_tok])
            tokens_str.append(ts)
            risk_vals.append(self._token_risk(ent, lp))

            # ── Decide if expensive signals needed ──
            # Only request hidden_states on periodic analysis steps (every LLD_COMPUTE_INTERVAL tokens)
            need_hidden = (step >= LLD_MIN_TOKENS and step % LLD_COMPUTE_INTERVAL == 0)

            # ── KV-cached forward (1 token) ──
            nt = torch.tensor([[next_tok]], device=self.model.device)
            out = self.model(
                input_ids=nt,
                past_key_values=past_kv,
                output_hidden_states=need_hidden,
                use_cache=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1:, :].squeeze(1)  # [1, vocab]

            # ── LLD (sparse, periodic, GPU) ──
            if need_hidden:
                cached_lld = min(self.compute_lld(out.hidden_states).lld, LLD_MAX)
            pt_lld.append(cached_lld)

            # ── RCD (computed on same periodic schedule as LLD) ──
            if need_hidden and step >= RCD_MIN_TOKENS:
                h = out.hidden_states[-1][:, -1, :].squeeze(0)
                cached_rcd, _ = self.compute_rcd(h)
                cached_rcd = min(cached_rcd, RCD_MAX)
            elif need_hidden and out.hidden_states is not None:
                # Seed previous hidden for first RCD computation
                self._prev_hidden = out.hidden_states[-1][:, -1, :].squeeze(0).clone().detach()
            pt_rcd.append(cached_rcd)

            # ── CSG (periodic, Stage 2 signal) ──
            if step >= SEMANTIC_MIN_TOKENS and step % CSG_COMPUTE_INTERVAL == 0:
                gen_so_far = "".join(tokens_str)
                csg_res = self.compute_csg(prompt, gen_so_far, _safe_mean(logprob_vals))
                cached_csg = min(csg_res.csg, CSG_MAX)
            pt_csg.append(cached_csg)

            # ── Running HRI (Stage 2: token signals only) ──
            m_ent = _safe_mean(entropy_vals)
            m_eng = _safe_mean(energy_vals)
            n_var = min(1.0, _safe_variance(logprob_vals) / 4.0)
            m_lld = _safe_mean(pt_lld) if step >= LLD_MIN_TOKENS else 0.0
            m_rcd = _safe_mean(pt_rcd)

            hri = self.compute_hri_dynamic(
                m_ent, m_eng, n_var, pt_csg[-1], m_lld, m_rcd,
                weights, prompt_risk_bias=prompt_risk_bias,
                debug=debug and step % 10 == 0,
            )
            hri_vals.append(hri)

            if debug and step % 10 == 0:
                print(f"[{step}] ent={m_ent:.3f} csg={pt_csg[-1]:.3f} "
                      f"lld={m_lld:.3f} rcd={m_rcd:.3f} → HRI={hri:.4f}")

            # ── Intervention ──
            if hri >= HRI_INTERVENTION_THRESHOLD and not intervention_triggered and step >= SEMANTIC_MIN_TOKENS:
                intervention_triggered = True
                intervention_msg = f"⚠️ HRI={hri:.2f} — possible hallucination detected."

            # ── Early stop (slope-gated) ──
            if early_stop_enabled and step >= EARLY_STOP_MIN_TOKENS and hri >= hri_threshold:
                sl = _compute_hri_slope(hri_vals, 4)
                if sl > 0:
                    early_stopped = True
                    if debug:
                        print(f"⚠️ Early stop step {step}: HRI={hri:.4f} slope={sl:.4f}")
                    break

            # EOS
            if next_tok == self.tokenizer.eos_token_id:
                break

            # Stop sequences
            cur = "".join(tokens_str)
            found = False
            for ss in STOP_SEQUENCES:
                if ss in cur:
                    si = cur.find(ss)
                    if si > 0:
                        keep = 0
                        acc = 0
                        for j, t in enumerate(tokens_str):
                            acc += len(t)
                            if acc > si:
                                keep = j
                                break
                            keep = j + 1
                        if 0 < keep < len(tokens_str):
                            tokens_str = tokens_str[:keep]
                            tokens_id = tokens_id[:keep]
                            entropy_vals = entropy_vals[:keep]
                            energy_vals = energy_vals[:keep]
                            logprob_vals = logprob_vals[:keep]
                            risk_vals = risk_vals[:keep]
                            hri_vals = hri_vals[:keep]
                            pt_lld = pt_lld[:keep]
                            pt_rcd = pt_rcd[:keep]
                            pt_csg = pt_csg[:keep]
                    found = True
                    break
            if found:
                break

        # ── Post-generation ──────────────────────────────────────────────────
        gen_text = "".join(tokens_str)
        for ss in STOP_SEQUENCES:
            if ss in gen_text:
                gen_text = gen_text.split(ss)[0]
        gen_text = gen_text.rstrip()

        if logprob_vals:
            final_csg = self.compute_csg(prompt, gen_text, _safe_mean(logprob_vals))
        else:
            final_csg = CSGResult()

        # V5: Post-hoc semantic fabrication signals (Stage 3)
        f_novelty = 1.0 - final_csg.semantic_stability
        f_entity = _compute_entity_expansion(prompt, gen_text)
        f_claim = _compute_claim_density(gen_text)

        ent_mean = _safe_mean(entropy_vals)
        ent_max = max(entropy_vals) if entropy_vals else 0.0
        eng_mean = _safe_mean(energy_vals)
        lp_var = _safe_variance(logprob_vals)
        lp_std = math.sqrt(lp_var) if lp_var > 0 else 0.0
        nll = -_safe_mean(logprob_vals) if logprob_vals else 0.0
        ppl = math.exp(nll) if nll < 20 else float("inf")
        f_lld = _safe_mean(pt_lld)
        f_rcd = _safe_mean(pt_rcd)
        slope, accel = self._hri_slope_accel(hri_vals)

        return GenerationResult(
            text=prompt + gen_text,
            prompt=prompt,
            tokens=tokens_str,
            token_risks=risk_vals,
            hri_values=hri_vals,
            early_stopped=early_stopped,
            intervention_triggered=intervention_triggered,
            intervention_message=intervention_msg,
            entropy_values=entropy_vals,
            energy_values=energy_vals,
            logprob_values=logprob_vals,
            entropy_mean=ent_mean,
            entropy_max=ent_max,
            semantic_energy_mean=eng_mean,
            logprob_variance=lp_var,
            logprob_std=lp_std,
            nll=nll,
            perplexity=ppl,
            confidence=final_csg.confidence,
            semantic_stability=final_csg.semantic_stability,
            confidence_stability_gap=final_csg.csg,
            csg_result=final_csg,
            logit_lens_divergence=f_lld,
            lld_result=LLDResult(lld=f_lld),
            per_token_lld=pt_lld,
            representation_collapse=f_rcd,
            rcd_result=RCDResult(collapse_score=f_rcd, per_token_collapse=pt_rcd, is_collapsing=f_rcd > 0.8),
            per_token_rcd=pt_rcd,
            per_token_csg=pt_csg,
            novelty=f_novelty,
            entity_expansion=f_entity,
            claim_density=f_claim,
            per_token_novelty=[],  # no longer tracked per-token in V5
            # V5: Prompt perplexity from prefill (Stage 1)
            prompt_perplexity=prompt_ppl,
            prompt_logprobs=prompt_logprobs_list,
            min_prompt_logprob=min_prompt_lp,
            low_prob_fraction=low_prob_frac,
            risk_breakdown={
                "entropy": ent_mean, "semantic_energy": eng_mean,
                "logprob_variance": min(1.0, lp_var / 4.0),
                "csg": final_csg.csg, "lld": f_lld, "rcd": f_rcd,
                "novelty": f_novelty, "entity_expansion": f_entity, "claim_density": f_claim,
                "confidence": final_csg.confidence, "semantic_stability": final_csg.semantic_stability,
            },
            hri_slope=slope,
            hri_acceleration=accel,
            vocab_size=self.vocab_size,
            input_ids=input_ids,
            generated_ids=None,
        )

    def analyze_generation(self, prompt: str, max_new_tokens: int = 60,
                           temperature: float = 0.7, debug: bool = False) -> Dict[str, Any]:
        r = self.generate(prompt=prompt, max_new_tokens=max_new_tokens,
                          temperature=temperature, debug=debug)
        hri = r.hri_values[-1] if r.hri_values else 0.0
        return {
            "hri": hri,
            "generated_text": r.text[len(prompt):],
            "signals": {
                "entropy": r.entropy_mean, "semantic_energy": r.semantic_energy_mean,
                "confidence": r.confidence, "semantic_stability": r.semantic_stability,
                "csg": r.confidence_stability_gap, "logit_lens_divergence": r.logit_lens_divergence,
                "representation_collapse": r.representation_collapse, "logprob_variance": r.logprob_variance,
                "novelty": r.novelty, "entity_expansion": r.entity_expansion, "claim_density": r.claim_density,
            },
            "per_token": {"hri": r.hri_values, "csg": r.per_token_csg, "lld": r.per_token_lld,
                          "rcd": r.per_token_rcd},
            "prompt_perplexity": r.prompt_perplexity,
            "early_stopped": r.early_stopped,
            "intervention_message": r.intervention_message,
        }
