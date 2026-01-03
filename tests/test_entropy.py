"""
Test Entropy Metrics Module.

Unit tests for TokenBurn-style token-level entropy computation.
"""

import pytest
import math
import torch

from halluscope.metrics.entropy import (
    token_entropies,
    aggregate_entropy,
    normalized_entropy,
    entropy_variance,
)


class TestTokenEntropies:
    """Tests for token_entropies function."""

    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution over vocab should have high entropy."""
        vocab_size = 100
        logits = torch.zeros(1, vocab_size)
        scores = [logits]

        entropies = token_entropies(scores)

        assert len(entropies) == 1
        expected = math.log(vocab_size)  # Max entropy
        assert abs(entropies[0] - expected) < 0.01

    def test_deterministic_distribution_zero_entropy(self):
        """One-hot distribution should have near-zero entropy."""
        vocab_size = 100
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 42] = 100.0
        scores = [logits]

        entropies = token_entropies(scores)

        assert len(entropies) == 1
        assert entropies[0] < 0.01

    def test_multiple_tokens(self):
        """Test entropy computation for multiple tokens."""
        vocab_size = 50
        uniform_logits = torch.zeros(1, vocab_size)
        peaked_logits = torch.full((1, vocab_size), -50.0)
        peaked_logits[0, 0] = 50.0

        scores = [uniform_logits, peaked_logits]
        entropies = token_entropies(scores)

        assert len(entropies) == 2
        assert entropies[0] > entropies[1]

    def test_empty_scores(self):
        """Empty scores list should return empty entropies."""
        entropies = token_entropies([])
        assert entropies == []


class TestAggregateEntropy:
    """Tests for aggregate_entropy function."""

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        entropies = [1.0, 2.0, 3.0]
        result = aggregate_entropy(entropies, mode="mean")
        assert result == pytest.approx(2.0)

    def test_max_aggregation(self):
        """Test max aggregation."""
        entropies = [1.0, 5.0, 3.0]
        result = aggregate_entropy(entropies, mode="max")
        assert result == 5.0

    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        result = aggregate_entropy([], mode="mean")
        assert result == 0.0

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            aggregate_entropy([1.0, 2.0], mode="invalid")


class TestNormalizedEntropy:
    """Tests for normalized_entropy function."""

    def test_max_entropy_returns_one(self):
        """Entropy = log(vocab_size) should give normalized = 1."""
        vocab_size = 1000
        entropy = math.log(vocab_size)
        result = normalized_entropy(entropy, vocab_size)
        assert result == pytest.approx(1.0)

    def test_zero_entropy_returns_zero(self):
        """Zero entropy should give normalized = 0."""
        result = normalized_entropy(0.0, 1000)
        assert result == 0.0

    def test_half_entropy(self):
        """Half of max entropy should give 0.5."""
        vocab_size = 1000
        entropy = math.log(vocab_size) / 2
        result = normalized_entropy(entropy, vocab_size)
        assert result == pytest.approx(0.5)

    def test_small_vocab(self):
        """Edge case: vocab_size = 1 should return 0."""
        result = normalized_entropy(0.0, 1)
        assert result == 0.0


class TestEntropyVariance:
    """Tests for entropy_variance function."""

    def test_constant_entropies_zero_variance(self):
        """All same values should have zero variance."""
        entropies = [3.0, 3.0, 3.0, 3.0]
        result = entropy_variance(entropies)
        assert result == pytest.approx(0.0)

    def test_varied_entropies(self):
        """Different values should have non-zero variance."""
        entropies = [1.0, 2.0, 3.0, 4.0]
        result = entropy_variance(entropies)
        # Population variance = mean((x - mean)^2)
        # mean = 2.5, variance = ((1.5)^2 + (0.5)^2 + (0.5)^2 + (1.5)^2) / 4 = 1.25
        assert result == pytest.approx(1.25)

    def test_single_value(self):
        """Single value should return 0."""
        result = entropy_variance([3.0])
        assert result == 0.0

    def test_empty_list(self):
        """Empty list should return 0."""
        result = entropy_variance([])
        assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
