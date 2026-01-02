"""
Test Entropy Metrics Module.

Unit tests for token-level entropy computation.
"""

import pytest
import torch

from halluscope.metrics.entropy import token_entropies, aggregate_entropy


class TestTokenEntropies:
    """Tests for token_entropies function."""

    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution over vocab should have high entropy."""
        vocab_size = 100
        # Uniform logits → equal probabilities
        logits = torch.zeros(1, vocab_size)
        scores = [logits]

        entropies = token_entropies(scores)

        assert len(entropies) == 1
        # Entropy of uniform distribution = log(vocab_size)
        expected = torch.log(torch.tensor(float(vocab_size))).item()
        assert abs(entropies[0] - expected) < 0.01

    def test_deterministic_distribution_zero_entropy(self):
        """One-hot distribution should have near-zero entropy."""
        vocab_size = 100
        # Very peaked logits → one token has ~1.0 probability
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 42] = 100.0  # Token 42 is certain
        scores = [logits]

        entropies = token_entropies(scores)

        assert len(entropies) == 1
        assert entropies[0] < 0.01  # Near zero

    def test_multiple_tokens(self):
        """Test entropy computation for multiple tokens."""
        vocab_size = 50
        # Two tokens with different entropy levels
        uniform_logits = torch.zeros(1, vocab_size)
        peaked_logits = torch.full((1, vocab_size), -50.0)
        peaked_logits[0, 0] = 50.0

        scores = [uniform_logits, peaked_logits]
        entropies = token_entropies(scores)

        assert len(entropies) == 2
        assert entropies[0] > entropies[1]  # Uniform has higher entropy

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

    def test_single_value(self):
        """Single value should return itself."""
        result = aggregate_entropy([3.5], mode="mean")
        assert result == 3.5

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            aggregate_entropy([1.0, 2.0], mode="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
