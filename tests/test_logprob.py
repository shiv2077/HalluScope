"""
Test Log-Probability Metrics Module.

Unit tests for TokenBurn-style NLL, perplexity, and variance computation.
"""

import pytest
import math
import torch

from halluscope.metrics.logprob import (
    token_logprobs,
    aggregate_nll,
    perplexity,
    logprob_variance,
    logprob_std,
)


class TestTokenLogprobs:
    """Tests for token_logprobs function."""

    def test_certain_token_zero_logprob(self):
        """Token with probability ~1 should have logprob ~0."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 42] = 100.0
        scores = [logits]
        new_token_ids = torch.tensor([42])

        logprobs = token_logprobs(scores, new_token_ids)

        assert len(logprobs) == 1
        assert logprobs[0] > -0.01

    def test_unlikely_token_negative_logprob(self):
        """Unlikely token should have large negative logprob."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 0] = 100.0
        scores = [logits]
        new_token_ids = torch.tensor([50])

        logprobs = token_logprobs(scores, new_token_ids)

        assert len(logprobs) == 1
        assert logprobs[0] < -10.0

    def test_uniform_distribution(self):
        """Uniform distribution logprob should equal -log(vocab_size)."""
        vocab_size = 100
        logits = torch.zeros(1, vocab_size)
        scores = [logits]
        new_token_ids = torch.tensor([42])

        logprobs = token_logprobs(scores, new_token_ids)

        expected = -math.log(vocab_size)
        assert abs(logprobs[0] - expected) < 0.01

    def test_empty_scores(self):
        """Empty scores should return empty logprobs."""
        logprobs = token_logprobs([], torch.tensor([]))
        assert logprobs == []


class TestAggregateNLL:
    """Tests for aggregate_nll function."""

    def test_nll_is_negative_mean_logprob(self):
        """NLL should be negative of mean logprob."""
        logprobs = [-1.0, -2.0, -3.0]
        nll = aggregate_nll(logprobs)
        assert nll == pytest.approx(2.0)

    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        nll = aggregate_nll([])
        assert nll == 0.0

    def test_zero_logprobs(self):
        """Zero logprobs should give zero NLL."""
        nll = aggregate_nll([0.0, 0.0, 0.0])
        assert nll == 0.0


class TestPerplexity:
    """Tests for perplexity function."""

    def test_perplexity_is_exp_nll(self):
        """Perplexity should equal exp(NLL)."""
        nll = 2.0
        ppl = perplexity(nll)
        assert ppl == pytest.approx(math.exp(2.0))

    def test_zero_nll_gives_one(self):
        """Zero NLL should give perplexity of 1."""
        ppl = perplexity(0.0)
        assert ppl == pytest.approx(1.0)

    def test_high_nll_high_perplexity(self):
        """High NLL should give high perplexity."""
        ppl = perplexity(5.0)
        assert ppl > 100  # exp(5) ≈ 148


class TestLogprobVariance:
    """Tests for logprob_variance function."""

    def test_constant_logprobs_zero_variance(self):
        """All same values should have zero variance."""
        logprobs = [-2.0, -2.0, -2.0, -2.0]
        result = logprob_variance(logprobs)
        assert result == pytest.approx(0.0)

    def test_varied_logprobs(self):
        """Different values should have non-zero variance."""
        logprobs = [-1.0, -2.0, -3.0, -4.0]
        result = logprob_variance(logprobs)
        # Population variance = 1.25
        assert result == pytest.approx(1.25)

    def test_single_value(self):
        """Single value should return 0."""
        result = logprob_variance([-2.0])
        assert result == 0.0

    def test_empty_list(self):
        """Empty list should return 0."""
        result = logprob_variance([])
        assert result == 0.0


class TestLogprobStd:
    """Tests for logprob_std function."""

    def test_std_is_sqrt_variance(self):
        """Std should be sqrt of variance."""
        logprobs = [-1.0, -2.0, -3.0, -4.0]
        var = logprob_variance(logprobs)
        std = logprob_std(logprobs)
        assert std == pytest.approx(math.sqrt(var))

    def test_zero_variance_zero_std(self):
        """Zero variance should give zero std."""
        logprobs = [-2.0, -2.0, -2.0]
        std = logprob_std(logprobs)
        assert std == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
