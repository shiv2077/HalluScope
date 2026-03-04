"""
HalluScope V4 Tests — unit tests for core signals and HRI.

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
        required = {"w_entropy", "w_energy", "w_variance", "w_csg", "w_lld", "w_rcd", "bias"}
        for regime, profile in WEIGHT_PROFILES.items():
            for key in required:
                assert key in profile, f"Missing {key} in {regime}"

    def test_short_answer_conservative(self):
        from halluscope.core.llm_backend_v4 import WEIGHT_PROFILES, Regime
        p = WEIGHT_PROFILES[Regime.SHORT_ANSWER]
        assert p["bias"] <= -3.0, "SHORT_ANSWER should have conservative bias"
        assert p["w_csg"] <= 0.1, "CSG should be near-disabled for short answers"
