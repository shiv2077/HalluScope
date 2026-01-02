"""
Test Hallucination Risk Index Module.

Unit tests for HRI-v1 computation and clarification triggers.
"""

import pytest
import math

from halluscope.risk.hri import (
    sigmoid,
    compute_hri_v1,
    explain_v1,
    clarification_questions,
)


class TestSigmoid:
    """Tests for sigmoid helper function."""

    def test_sigmoid_at_zero(self):
        """sigmoid(0) = 0.5."""
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        """sigmoid(large positive) ≈ 1."""
        assert sigmoid(10.0) > 0.999

    def test_sigmoid_large_negative(self):
        """sigmoid(large negative) ≈ 0."""
        assert sigmoid(-10.0) < 0.001

    def test_sigmoid_symmetry(self):
        """sigmoid(x) + sigmoid(-x) = 1."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)


class TestComputeHriV1:
    """Tests for compute_hri_v1 function."""

    def test_default_weights(self):
        """Test HRI-v1 with default weights (0.5, 0.5, -2.0)."""
        # entropy_mean=0, nll=0 → x = 0.5*0 + 0.5*0 - 2.0 = -2.0
        # sigmoid(-2.0) ≈ 0.119
        hri = compute_hri_v1(entropy_mean=0.0, nll=0.0)
        expected = 1 / (1 + math.exp(2.0))  # ~0.119
        assert hri == pytest.approx(expected, rel=0.01)

    def test_high_entropy_high_nll(self):
        """High entropy + high NLL should give HRI > 0.5."""
        # entropy_mean=4, nll=4 → x = 0.5*4 + 0.5*4 - 2.0 = 2.0
        # sigmoid(2.0) ≈ 0.88
        hri = compute_hri_v1(entropy_mean=4.0, nll=4.0)
        assert hri > 0.8

    def test_moderate_values_near_threshold(self):
        """Test values around the 0.65 threshold."""
        # Find (entropy, nll) such that HRI ≈ 0.65
        # sigmoid(x) = 0.65 → x = ln(0.65/0.35) ≈ 0.62
        # 0.5*e + 0.5*n - 2 = 0.62 → e+n ≈ 5.24
        hri = compute_hri_v1(entropy_mean=2.62, nll=2.62)
        assert 0.6 < hri < 0.7

    def test_custom_weights(self):
        """Test with custom weights."""
        hri = compute_hri_v1(entropy_mean=1.0, nll=1.0, wH=1.0, wN=0.0, b=0.0)
        # x = 1.0*1 + 0.0*1 + 0 = 1.0
        expected = 1 / (1 + math.exp(-1.0))  # ~0.731
        assert hri == pytest.approx(expected, rel=0.01)

    def test_bounded_between_0_and_1(self):
        """HRI should always be in [0, 1]."""
        test_cases = [
            (0.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (10.0, 0.0),
            (100.0, 100.0),
        ]
        for ent, nll in test_cases:
            hri = compute_hri_v1(entropy_mean=ent, nll=nll)
            assert 0.0 <= hri <= 1.0


class TestExplainV1:
    """Tests for explain_v1 function."""

    def test_low_uncertainty_default_message(self):
        """Low entropy and NLL should return default 'within bounds' message."""
        explanation = explain_v1(entropy_mean=1.0, nll=1.0)
        assert len(explanation) >= 1
        # Should be the default message
        assert "normal" in explanation[0].lower() or "bounds" in explanation[0].lower()

    def test_high_entropy_explanation(self):
        """High entropy should produce entropy-related note."""
        explanation = explain_v1(entropy_mean=5.5, nll=1.0)
        # Should have entropy-related explanation
        assert any("entropy" in note.lower() for note in explanation)

    def test_high_nll_explanation(self):
        """High NLL should produce NLL-related note."""
        explanation = explain_v1(entropy_mean=1.0, nll=4.5)
        # Should have NLL-related explanation
        assert any("nll" in note.lower() for note in explanation)

    def test_both_high_multiple_notes(self):
        """Both high entropy and NLL should produce multiple notes."""
        explanation = explain_v1(entropy_mean=5.5, nll=4.5)
        assert len(explanation) >= 2


class TestClarificationQuestions:
    """Tests for clarification_questions function."""

    def test_returns_list(self):
        """Should return a list of questions."""
        questions = clarification_questions()
        assert isinstance(questions, list)
        assert len(questions) >= 2

    def test_questions_are_strings(self):
        """All questions should be non-empty strings."""
        questions = clarification_questions()
        for q in questions:
            assert isinstance(q, str)
            assert len(q) > 0

    def test_questions_are_questions(self):
        """Questions should end with question mark or be phrased as questions."""
        questions = clarification_questions()
        for q in questions:
            # Either ends with ? or contains typical question words
            assert q.endswith("?") or any(
                word in q.lower() for word in ["what", "can", "do", "how", "which"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
