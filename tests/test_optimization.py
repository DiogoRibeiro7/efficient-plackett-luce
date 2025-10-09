"""
Tests for optimization functions.
"""

import numpy as np

from plackett_luce.optimization import (
    compute_convergence,
    compute_log_likelihood_pl,
    newman_iteration_full_pl,
    normalize_scores,
)


class TestOptimizationFunctions:
    """Test suite for Numba-optimized functions."""

    def test_normalize_scores(self):
        """Test score normalization."""
        scores = np.array([1.0, 2.0, 4.0, 8.0])
        normalized = normalize_scores(scores)

        # Geometric mean should be 1
        geometric_mean = np.exp(np.mean(np.log(normalized)))
        assert np.isclose(geometric_mean, 1.0)

        # Should preserve relative ordering
        assert np.all(normalized[:-1] <= normalized[1:]) or np.all(
            normalized[:-1] >= normalized[1:]
        )

    def test_compute_convergence(self):
        """Test convergence metric computation."""
        old_scores = np.array([1.0, 2.0, 3.0])
        new_scores = np.array([1.1, 2.0, 2.9])

        A = compute_convergence(old_scores, new_scores)

        assert isinstance(A, float)
        assert A >= 0

        # Same scores should give 0 convergence
        A_same = compute_convergence(old_scores, old_scores)
        assert np.isclose(A_same, 0.0)

    def test_newman_iteration(self):
        """Test Newman's iteration step."""
        N = 3
        scores = np.array([1.0, 2.0, 3.0])

        # Simple edge: [0, 1, 2]
        edges = np.array([0, 1, 2], dtype=np.int32)
        weights = np.array([1.0])
        edge_lengths = np.array([3], dtype=np.int32)
        edge_starts = np.array([0], dtype=np.int32)

        new_scores = newman_iteration_full_pl(scores, edges, weights, edge_lengths, edge_starts)

        assert len(new_scores) == N
        assert np.all(new_scores > 0)

    def test_log_likelihood_computation(self):
        """Test log-likelihood computation."""
        scores = np.array([1.0, 2.0, 3.0])

        # Edge: [2, 1, 0] (node 2 > node 1 > node 0)
        edges = np.array([2, 1, 0], dtype=np.int32)
        weights = np.array([1.0])
        edge_lengths = np.array([3], dtype=np.int32)
        edge_starts = np.array([0], dtype=np.int32)

        ll = compute_log_likelihood_pl(scores, edges, weights, edge_lengths, edge_starts)

        assert isinstance(ll, float)
        assert ll < 0  # Log-likelihood should be negative
