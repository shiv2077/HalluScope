"""
Test Log-Probability Metrics Module.

Unit tests for NLL and log-probability computation.
"""

import pytest
import torch

from halluscope.metrics.logprob import token_logprobs, aggregate_nll


class TestTokenLogprobs:
    """Tests for token_logprobs function."""

    def test_certain_token_zero_logprob(self):
        """Token with probability ~1 should have logprob ~0."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 42] = 100.0  # Token 42 has prob ~1
        scores = [logits]
        new_token_ids = [42]

        logprobs = token_logprobs(scores, new_token_ids)

        assert len(logprobs) == 1
        assert logprobs[0] > -0.01  # Very close to 0

    def test_unlikely_token_negative_logprob(self):
        """Unlikely token should have large negative logprob."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 0] = 100.0  # Token 0 is certain
        scores = [logits]
        new_token_ids = [50]  # But we sample token 50 (unlikely)

        logprobs = token_logprobs(scores, new_token_ids)

        assert len(logprobs) == 1
        assert logprobs[0] < -10.0  # Very negative

    def test_uniform_distribution(self):
        """Uniform distribution logprob should equal -log(vocab_size)."""
        vocab_size = 100
        logits = torch.zeros(1, vocab_size)  # Uniform
        scores = [logits]
        new_token_ids = [42]

        logprobs = token_logprobs(scores, new_token_ids)

        expected = -torch.log(torch.tensor(float(vocab_size))).item()
        assert abs(logprobs[0] - expected) < 0.01

    def test_multiple_tokens(self):
        """Test logprobs for multiple tokens."""
        vocab_size = 50
        logits1 = torch.zeros(1, vocab_size)
        logits2 = torch.zeros(1, vocab_size)
        scores = [logits1, logits2]
        new_token_ids = [10, 20]

        logprobs = token_logprobs(scores, new_token_ids)

        assert len(logprobs) == 2
        # Both should have same logprob (uniform distribution)
        assert abs(logprobs[0] - logprobs[1]) < 0.001

    def test_empty_scores(self):
        """Empty scores should return empty logprobs."""
        logprobs = token_logprobs([], [])
        assert logprobs == []


class TestAggregateNLL:
    """Tests for aggregate_nll function."""

    def test_nll_is_negative_mean_logprob(self):
        """NLL should be negative of mean logprob."""
        logprobs = [-1.0, -2.0, -3.0]
        nll = aggregate_nll(logprobs)
        # Mean logprob = -2.0, so NLL = 2.0
        assert nll == pytest.approx(2.0)

    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        nll = aggregate_nll([])
        assert nll == 0.0

    def test_single_value(self):
        """Single value NLL."""
        nll = aggregate_nll([-3.5])
        assert nll == pytest.approx(3.5)

    def test_zero_logprobs(self):
        """Zero logprobs (certain tokens) should give zero NLL."""
        nll = aggregate_nll([0.0, 0.0, 0.0])
        assert nll == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
