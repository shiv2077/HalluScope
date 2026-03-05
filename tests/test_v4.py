"""
HalluScope V5 Tests — unit tests for core signals and HRI.

These tests validate the mathematical signal computation without
loading any ML models (pure unit tests).
"""

import math
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test HRI classification (hri.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyRisk:
    def test_low(self):
        from halluscope.risk.hri import classify_risk
        assert classify_risk(0.0) == "LOW"
        assert classify_risk(0.20) == "LOW"
        assert classify_risk(0.44) == "LOW"

    def test_medium(self):
        from halluscope.risk.hri import classify_risk
        assert classify_risk(0.45) == "MEDIUM"
        assert classify_risk(0.55) == "MEDIUM"
        assert classify_risk(0.69) == "MEDIUM"

    def test_high(self):
        from halluscope.risk.hri import classify_risk
        assert classify_risk(0.70) == "HIGH"
        assert classify_risk(0.85) == "HIGH"
        assert classify_risk(1.0) == "HIGH"


class TestSigmoid:
    def test_zero(self):
        from halluscope.risk.hri import sigmoid
        assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_large_positive(self):
        from halluscope.risk.hri import sigmoid
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        from halluscope.risk.hri import sigmoid
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        from halluscope.risk.hri import sigmoid
        assert sigmoid(2.0) + sigmoid(-2.0) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test utility functions (llm_backend_v4.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_safe_mean(self):
        from halluscope.core.llm_backend_v4 import _safe_mean
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert _safe_mean([]) == 0.0
        assert _safe_mean([float("nan"), 1.0, 2.0]) == pytest.approx(1.5)
        assert _safe_mean([float("inf"), 1.0]) == pytest.approx(1.0)

    def test_safe_variance(self):
        from halluscope.core.llm_backend_v4 import _safe_variance
        assert _safe_variance([1.0]) == 0.0  # single element
        assert _safe_variance([]) == 0.0
        assert _safe_variance([1.0, 1.0, 1.0]) == pytest.approx(0.0)
        assert _safe_variance([0.0, 2.0]) == pytest.approx(1.0)

    def test_sigmoid(self):
        from halluscope.core.llm_backend_v4 import _sigmoid
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)
        assert _sigmoid(10.0) > 0.99
        assert _sigmoid(-10.0) < 0.01

    def test_compute_hri_slope(self):
        from halluscope.core.llm_backend_v4 import _compute_hri_slope
        # Flat
        assert _compute_hri_slope([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
        # Rising
        assert _compute_hri_slope([0.1, 0.2, 0.3, 0.4]) > 0
        # Falling
        assert _compute_hri_slope([0.4, 0.3, 0.2, 0.1]) < 0
        # Too few
        assert _compute_hri_slope([0.5]) == 0.0


class TestRegimeDetection:
    """Test regime detection with a mock tokenizer."""

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    def test_short_answer(self):
        from halluscope.core.llm_backend_v4 import _detect_regime, Regime
        tok = self.MockTokenizer()
        assert _detect_regime("what is 2+2", tok) == Regime.SHORT_ANSWER  # 3 tokens
        assert _detect_regime("hi there", tok) == Regime.SHORT_ANSWER  # 2 tokens

    def test_factual(self):
        from halluscope.core.llm_backend_v4 import _detect_regime, Regime
        tok = self.MockTokenizer()
        prompt = " ".join(["word"] * 15)  # 15 tokens
        assert _detect_regime(prompt, tok) == Regime.FACTUAL

    def test_generative(self):
        from halluscope.core.llm_backend_v4 import _detect_regime, Regime
        tok = self.MockTokenizer()
        prompt = " ".join(["word"] * 25)  # 25 tokens
        assert _detect_regime(prompt, tok) == Regime.GENERATIVE


# ─────────────────────────────────────────────────────────────────────────────
# Test CSGResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestCSGResult:
    def test_defaults(self):
        from halluscope.core.llm_backend_v4 import CSGResult
        r = CSGResult()
        assert r.confidence == 0.0
        assert r.semantic_stability == 0.0
        assert r.csg == 0.0

    def test_csg_formula(self):
        """CSG = confidence * (1 - stability)"""
        from halluscope.core.llm_backend_v4 import CSGResult
        r = CSGResult(confidence=0.9, semantic_stability=0.3, csg=0.9 * (1 - 0.3))
        assert r.csg == pytest.approx(0.63)


class TestWeightProfiles:
    def test_all_profiles_exist(self):
        from halluscope.core.llm_backend_v4 import WEIGHT_PROFILES, Regime
        assert Regime.SHORT_ANSWER in WEIGHT_PROFILES
        assert Regime.FACTUAL in WEIGHT_PROFILES
        assert Regime.GENERATIVE in WEIGHT_PROFILES

    def test_all_weights_present(self):
        from halluscope.core.llm_backend_v4 import WEIGHT_PROFILES
        # V5: Stage 2 only — token-level signals (no novelty/entity/claim)
        required = {"w_entropy", "w_energy", "w_variance", "w_csg", "w_lld", "w_rcd", "bias"}
        for regime, profile in WEIGHT_PROFILES.items():
            for key in required:
                assert key in profile, f"Missing {key} in {regime}"

    def test_short_answer_conservative(self):
        from halluscope.core.llm_backend_v4 import WEIGHT_PROFILES, Regime
        p = WEIGHT_PROFILES[Regime.SHORT_ANSWER]
        assert p["bias"] <= -2.0, "SHORT_ANSWER should have conservative bias"
        assert p["w_csg"] <= 0.20, "CSG should be moderate for short answers"
        assert p["w_entropy"] >= 0.3, "Entropy should dominate short answers"


# ─────────────────────────────────────────────────────────────────────────────
# Test V4.2 Semantic Fabrication Signals
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityExpansion:
    def test_identical(self):
        from halluscope.core.llm_backend_v4 import _compute_entity_expansion
        assert _compute_entity_expansion("hello world", "hello world") == 0.0

    def test_novel_content(self):
        from halluscope.core.llm_backend_v4 import _compute_entity_expansion
        result = _compute_entity_expansion(
            "Who discovered penicillin",
            "Alexander Fleming discovered penicillin in 1928 at St Mary Hospital London"
        )
        assert result > 0.3, "Answer introduces many new content words"

    def test_fabrication_high(self):
        from halluscope.core.llm_backend_v4 import _compute_entity_expansion
        result = _compute_entity_expansion(
            "Summarize the paper by Henri Vallon",
            "Gravitational harmonics fluid dynamics wave propagation frequency amplitude theoretical framework quantum mechanics"
        )
        assert result > 0.5, "Fabricated content should have high entity expansion"

    def test_empty_gen(self):
        from halluscope.core.llm_backend_v4 import _compute_entity_expansion
        assert _compute_entity_expansion("hello world", "") == 0.0


class TestClaimDensity:
    def test_empty(self):
        from halluscope.core.llm_backend_v4 import _compute_claim_density
        assert _compute_claim_density("") == 0.0

    def test_no_patterns(self):
        from halluscope.core.llm_backend_v4 import _compute_claim_density
        assert _compute_claim_density("The cat sat on the mat.") == 0.0

    def test_high_assertions(self):
        from halluscope.core.llm_backend_v4 import _compute_claim_density
        result = _compute_claim_density(
            "The paper demonstrated a novel approach. The study proposed a new framework. "
            "This work introduced key innovations."
        )
        assert result > 0.3, "Multiple assertions should yield high claim density"

    def test_with_evidence(self):
        from halluscope.core.llm_backend_v4 import _compute_claim_density
        result = _compute_claim_density(
            "The study demonstrated new findings, published in the journal Nature. "
            "However, it is unclear whether the experiment can be replicated."
        )
        assert result < 0.3, "Evidence and hedging should reduce claim density"


# ─────────────────────────────────────────────────────────────────────────────
# V5 Stage 1 — Pre-Hoc Prompt Risk
# ─────────────────────────────────────────────────────────────────────────────

class TestPreHocPatterns:
    def test_citation_detected(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns
        r = analyze_prompt_patterns(
            "Summarize the paper: Quantum-Enhanced Blockchain by Dr. Elena Vasquez"
        )
        assert r.citation_score > 0.3, "Citation pattern should be detected"

    def test_no_citation(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns
        r = analyze_prompt_patterns("What is 2+2?")
        assert r.citation_score == 0.0, "No citation in simple math"

    def test_entity_density(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns
        r = analyze_prompt_patterns(
            "Dr. John Smith and Prof. Maria Garcia published in Nature 2024"
        )
        assert r.entity_density > 0.1, "Named entities should increase density"

    def test_temporal_risk(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns
        r = analyze_prompt_patterns("What happened in 2027?")
        assert r.temporal_risk > 0.0, "Future date should trigger temporal risk"

    def test_quoted_title(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns
        r = analyze_prompt_patterns('Summarize the paper "Neural Quantum Computing"')
        assert r.quoted_title_score > 0.0, "Quoted title should be detected"

    def test_compute_prompt_risk(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns, compute_prompt_risk
        pattern = analyze_prompt_patterns(
            "Summarize the paper: Quantum-Enhanced Blockchain by Dr. Elena Vasquez"
        )
        result = compute_prompt_risk(
            pattern_result=pattern,
            prompt_perplexity=80.0,  # High perplexity
            min_token_logprob=-8.0,  # Very surprising token
            low_prob_fraction=0.3,
        )
        assert result.prompt_risk_score > 0.3, "Risky prompt should score high"
        assert result.prompt_perplexity == 80.0

    def test_safe_prompt_low_risk(self):
        from halluscope.metrics.pre_hoc import analyze_prompt_patterns, compute_prompt_risk
        pattern = analyze_prompt_patterns("What is the capital of France?")
        result = compute_prompt_risk(
            pattern_result=pattern,
            prompt_perplexity=5.0,
            min_token_logprob=-2.0,
            low_prob_fraction=0.0,
        )
        assert result.prompt_risk_score < 0.3, "Safe prompt should score low"


# ─────────────────────────────────────────────────────────────────────────────
# V5 Stage 3 — Post-Hoc Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestPostHocVerification:
    def test_claim_density(self):
        from halluscope.metrics.post_hoc import compute_claim_density
        density, n_assert, n_evidence = compute_claim_density(
            "The study demonstrated a novel approach. This work introduced key findings."
        )
        assert density > 0.2, "Multiple assertions should yield claim density"
        assert n_assert > 0

    def test_confidence_paradox(self):
        from halluscope.metrics.post_hoc import compute_confidence_paradox
        # High confidence + high novelty = paradox
        assert compute_confidence_paradox(0.9, 0.8) > 0.6
        # Low confidence + low novelty = safe
        assert compute_confidence_paradox(0.2, 0.1) < 0.1

    def test_narrative_template(self):
        from halluscope.metrics.post_hoc import compute_narrative_template_score
        score, templates = compute_narrative_template_score(
            "The study demonstrated a novel approach that improves performance. "
            "Key findings include significant breakthroughs in the field."
        )
        assert score > 0.0, "Fabrication template language should be detected"
        assert len(templates) > 0

    def test_concept_explosion(self):
        from halluscope.metrics.post_hoc import compute_concept_explosion
        score = compute_concept_explosion(
            "cat",
            "The quantum mechanics of neural networks in blockchain "
            "technology applied to genomic sequencing of deep learning models."
        )
        assert score > 0.3, "Many new concepts should trigger explosion"

    def test_verify_integration(self):
        from halluscope.metrics.post_hoc import verify
        result = verify(
            prompt="What is 2+2?",
            answer="4",
            confidence=0.95,
            semantic_stability=0.90,
        )
        assert result.verification_risk < 0.5, "Simple factual answer should be low risk"

    def test_verify_fabrication(self):
        from halluscope.metrics.post_hoc import verify
        result = verify(
            prompt="Summarize the paper by Henri Vallon",
            answer=(
                "The paper demonstrated a groundbreaking approach to quantum "
                "gravitational harmonics, introducing a novel framework for "
                "wave propagation analysis that fundamentally transforms our "
                "understanding of fluid dynamics at the molecular level."
            ),
            confidence=0.85,
            semantic_stability=0.70,
        )
        # Without embeddings, novelty=0 so paradox=0, but claim density + templates catch it
        assert result.verification_risk > 0.3, "Fabricated content should score higher"
        assert result.claim_density > 0.3, "Fabricated content has high claim density"
        assert result.narrative_template_score > 0.0, "Fabrication templates present"
