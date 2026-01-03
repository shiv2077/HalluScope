"""
Test HRI (Hallucination Risk Index) Module.

Unit tests for TokenBurn-style HRI computation and risk classification.
"""

import pytest
import math

from halluscope.risk.hri import (
    compute_hri,
    classify_risk,
    compute_hri_with_label,
    explain,
    get_clarification_questions,
    DEFAULT_WEIGHTS,
    THRESHOLD_LOW,
    THRESHOLD_HIGH,
)


class TestComputeHRI:
    """Tests for compute_hri function."""

    def test_all_zero_inputs_uses_bias(self):
        """With all zero inputs, HRI is sigmoid(bias)."""
        score = compute_hri(
            entropy_mean=0.0,
            nll=0.0,
            logprob_std=0.0,
            kl_uniform=0.0,
        )
        expected = 1 / (1 + math.exp(2.0))  # bias is -2.0
        assert score == pytest.approx(expected, abs=0.01)

    def test_high_entropy_increases_risk(self):
        """Higher entropy should increase HRI."""
        low = compute_hri(entropy_mean=1.0, nll=2.0, logprob_std=0.5)
        high = compute_hri(entropy_mean=5.0, nll=2.0, logprob_std=0.5)
        assert high > low

    def test_high_nll_increases_risk(self):
        """Higher NLL should increase HRI."""
        low = compute_hri(entropy_mean=2.0, nll=1.0, logprob_std=0.5)
        high = compute_hri(entropy_mean=2.0, nll=5.0, logprob_std=0.5)
        assert high > low

    def test_high_std_increases_risk(self):
        """Higher logprob_std should increase HRI."""
        low = compute_hri(entropy_mean=2.0, nll=2.0, logprob_std=0.5)
        high = compute_hri(entropy_mean=2.0, nll=2.0, logprob_std=3.0)
        assert high > low

    def test_kl_zero_weight_by_default(self):
        """KL divergence should not affect HRI with default weights."""
        without_kl = compute_hri(entropy_mean=2.0, nll=2.0, logprob_std=1.0, kl_uniform=0.0)
        with_kl = compute_hri(entropy_mean=2.0, nll=2.0, logprob_std=1.0, kl_uniform=5.0)
        assert with_kl == pytest.approx(without_kl, abs=0.01)

    def test_kl_nonzero_weight(self):
        """KL divergence should affect HRI when weight is non-zero."""
        # Use actual parameter names: wK for kl weight
        without_kl = compute_hri(entropy_mean=2.0, nll=2.0, logprob_std=1.0, kl_uniform=0.0, wK=0.2)
        with_kl = compute_hri(entropy_mean=2.0, nll=2.0, logprob_std=1.0, kl_uniform=3.0, wK=0.2)
        # With nonzero wK, KL affects score (lower KL = higher risk due to subtraction)
        assert with_kl != without_kl

    def test_hri_bounded_zero_to_one(self):
        """HRI should always be between 0 and 1."""
        # Extreme low values
        low = compute_hri(entropy_mean=-100.0, nll=-100.0, logprob_std=0.0)
        assert 0 <= low <= 1

        # Extreme high values
        high = compute_hri(entropy_mean=100.0, nll=100.0, logprob_std=100.0)
        assert 0 <= high <= 1


class TestClassifyRisk:
    """Tests for classify_risk function."""

    def test_low_risk_below_threshold(self):
        """Scores below 0.45 should be LOW."""
        assert classify_risk(0.0) == "LOW"
        assert classify_risk(0.30) == "LOW"
        assert classify_risk(0.44) == "LOW"

    def test_medium_risk_in_range(self):
        """Scores 0.45 to 0.70 should be MEDIUM."""
        assert classify_risk(0.45) == "MEDIUM"
        assert classify_risk(0.55) == "MEDIUM"
        assert classify_risk(0.69) == "MEDIUM"

    def test_high_risk_above_threshold(self):
        """Scores >= 0.70 should be HIGH."""
        assert classify_risk(0.70) == "HIGH"
        assert classify_risk(0.85) == "HIGH"
        assert classify_risk(1.0) == "HIGH"

    def test_threshold_constants(self):
        """Verify threshold constants."""
        assert THRESHOLD_LOW == 0.45
        assert THRESHOLD_HIGH == 0.70


class TestComputeHRIWithLabel:
    """Tests for compute_hri_with_label function."""

    def test_returns_tuple(self):
        """Should return tuple with score and label."""
        result = compute_hri_with_label(entropy_mean=2.0, nll=2.0, logprob_std=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, label = result
        assert isinstance(score, float)
        assert isinstance(label, str)

    def test_label_matches_score(self):
        """Label should match classify_risk(score)."""
        score, label = compute_hri_with_label(entropy_mean=0.5, nll=0.5, logprob_std=0.3)
        assert label == classify_risk(score)


class TestExplain:
    """Tests for explain function."""

    def test_returns_list(self):
        """Should return explanation list."""
        explanations = explain(
            entropy_mean=3.0,
            nll=2.5,
            ppl=12.2,
            logprob_std=1.2,
            logprob_var=1.44,
            kl_uniform=0.5,
        )
        assert isinstance(explanations, list)
        assert len(explanations) > 0

    def test_all_strings(self):
        """All explanations should be strings."""
        explanations = explain(
            entropy_mean=5.5, nll=4.5, ppl=90.0, logprob_std=2.5, logprob_var=6.25
        )
        assert all(isinstance(e, str) for e in explanations)


class TestClarificationQuestions:
    """Tests for get_clarification_questions function."""

    def test_low_risk_none(self):
        """LOW risk should return None."""
        questions = get_clarification_questions("LOW")
        assert questions is None

    def test_medium_risk_has_questions(self):
        """MEDIUM risk should return 1+ questions."""
        questions = get_clarification_questions("MEDIUM")
        assert questions is not None
        assert len(questions) >= 1
        assert all(isinstance(q, str) for q in questions)

    def test_high_risk_has_questions(self):
        """HIGH risk should return questions."""
        questions = get_clarification_questions("HIGH")
        assert questions is not None
        assert len(questions) >= 2


class TestDefaultWeights:
    """Tests for DEFAULT_WEIGHTS constant."""

    def test_weights_sum_approximately_one(self):
        """Non-bias weights should sum to approximately 1."""
        total = (
            DEFAULT_WEIGHTS["wH"]
            + DEFAULT_WEIGHTS["wN"]
            + DEFAULT_WEIGHTS["wV"]
            + DEFAULT_WEIGHTS["wK"]
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_kl_weight_is_zero(self):
        """KL weight should be 0 by default."""
        assert DEFAULT_WEIGHTS["wK"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
