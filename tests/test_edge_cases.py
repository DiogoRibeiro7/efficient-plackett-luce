"""
Tests for edge cases, ties, and error handling.
"""

import numpy as np
import pytest

from plackett_luce import PlackettLuceModel, ProjectedPlackettLuce
from plackett_luce.utils import generate_synthetic_rankings


class TestEdgeCases:
    """Test edge cases in model behavior."""

    def test_single_comparison(self):
        """Test with only one comparison."""
        data = [(("A", "B"), 1)]

        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 2
        assert ranking[0][0] == "A"  # A beats B

    def test_two_node_tournament(self):
        """Test with only two nodes."""
        data = [(("A", "B"), 5), (("B", "A"), 2)]

        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert ranking[0][0] == "A"  # A wins more often

    def test_complete_dominance(self):
        """Test when one node always wins."""
        data = [
            (("A", "B"), 1),
            (("A", "C"), 1),
            (("A", "D"), 1),
            (("B", "C"), 1),
            (("B", "D"), 1),
            (("C", "D"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert ranking[0][0] == "A"
        assert ranking[-1][0] == "D"

    def test_circular_rankings(self):
        """Test with circular dominance (A>B, B>C, C>A)."""
        data = [
            (("A", "B"), 1),
            (("B", "C"), 1),
            (("C", "A"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should still converge and produce rankings
        assert model.is_fitted
        assert stats["iterations"] < model.max_iterations

        ranking = model.get_ranking()
        assert len(ranking) == 3

    def test_all_equal_performance(self):
        """Test when all nodes perform equally."""
        data = [
            (("A", "B"), 1),
            (("B", "A"), 1),
            (("A", "C"), 1),
            (("C", "A"), 1),
            (("B", "C"), 1),
            (("C", "B"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        scores = [score for _, score in ranking]

        # All scores should be approximately equal
        assert np.allclose(scores, scores[0], atol=1e-2)

    def test_single_node_in_data(self):
        """Test error handling with single node."""
        # This should fail since we need at least 2 nodes
        data = [(("A",), 1)]

        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit(data, verbose=False)

    def test_duplicate_nodes_in_ranking(self):
        """Test that duplicate nodes in ranking are rejected."""
        data = [(("A", "A", "B"), 1)]

        model = PlackettLuceModel(model_type="full")
        with pytest.raises(ValueError, match="duplicate nodes"):
            model.fit(data, verbose=False)

    def test_very_large_comparison(self):
        """Test with very large single comparison."""
        nodes = [f"Node{i}" for i in range(50)]
        data = [(tuple(nodes), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert model.is_fitted
        assert len(model.scores) == 50

    def test_many_small_comparisons(self):
        """Test with many pairwise comparisons."""
        data = [(("A", "B"), 1) for _ in range(100)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should handle repeated identical comparisons
        assert model.is_fitted

    def test_all_different_nodes(self):
        """Test when no nodes appear in multiple comparisons."""
        data = [
            (("A", "B"), 1),
            (("C", "D"), 1),
            (("E", "F"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 6


class TestTieHandling:
    """Test handling of ties and special ranking scenarios."""

    def test_identical_rankings(self):
        """Test with identical rankings."""
        data = [(("A", "B", "C"), 10)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should strongly favor A > B > C
        ranking = model.get_ranking()
        assert ranking[0][0] == "A"
        assert ranking[1][0] == "B"
        assert ranking[2][0] == "C"

    def test_conflicting_rankings(self):
        """Test with conflicting ranking information."""
        data = [
            (("A", "B", "C"), 5),
            (("C", "B", "A"), 5),  # Exact opposite
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # B should be middle, A and C should be close
        ranking = model.get_ranking()
        assert ranking[1][0] == "B"

    def test_sparse_connectivity(self):
        """Test with sparse node connectivity."""
        # Two separate groups with one link
        data = [
            (("A", "B"), 1),
            (("B", "C"), 1),
            (("D", "E"), 1),
            (("E", "F"), 1),
            (("C", "D"), 1),  # Only link between groups
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert model.is_fitted
        assert len(model.scores) == 6

    def test_perfect_tie_resolution(self):
        """Test that model can handle perfect ties."""
        # A and B always tie
        data = [
            (("A", "B", "C"), 1),
            (("B", "A", "C"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        score_a = model.get_score("A")
        score_b = model.get_score("B")

        # A and B should have similar scores
        assert np.isclose(score_a, score_b, rtol=0.1)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_empty_ranking(self):
        """Test error with empty ranking."""
        data = [((), 1)]

        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit(data, verbose=False)

    def test_negative_weight(self):
        """Test error with negative weight."""
        data = [(("A", "B"), -1)]

        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit(data, verbose=False)

    def test_zero_weight(self):
        """Test error with zero weight."""
        data = [(("A", "B"), 0)]

        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit(data, verbose=False)

    def test_non_tuple_ranking(self):
        """Test error with non-tuple ranking."""
        data = [("AB", 1)]  # String instead of tuple

        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit(data, verbose=False)

    def test_invalid_hyperedge_format(self):
        """Test error with invalid hyperedge format."""
        data = [("A", "B", "C")]  # Missing weight

        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit(data, verbose=False)

    def test_predict_before_fit(self):
        """Test error when predicting before fitting."""
        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.predict_probability(("A", "B"))

    def test_log_likelihood_before_fit(self):
        """Test error when computing log-likelihood before fitting."""
        model = PlackettLuceModel(model_type="full")
        data = [(("A", "B"), 1)]

        with pytest.raises(ValueError):
            model.log_likelihood(data)

    def test_get_ranking_before_fit(self):
        """Test error when getting ranking before fitting."""
        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.get_ranking()

    def test_get_score_unknown_node(self):
        """Test error when getting score for unknown node."""
        data = [(("A", "B"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        with pytest.raises(ValueError):
            model.get_score("Z")

    def test_save_before_fit(self):
        """Test error when saving before fitting."""
        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.save("model.pkl")

    def test_duplicate_nodes_in_ranking(self):
        """Test handling of duplicate nodes in ranking."""
        data = [(("A", "A", "B"), 1)]  # A appears twice

        model = PlackettLuceModel(model_type="full")
        # This might be accepted or rejected depending on implementation
        # If accepted, should handle gracefully
        try:
            model.fit(data, verbose=False)
            # If it works, verify it doesn't crash
            ranking = model.get_ranking()
            assert len(ranking) >= 2
        except ValueError:
            # Also acceptable to reject
            pass


class TestNumericalStability:
    """Test numerical stability and extreme values."""

    def test_very_large_weights(self):
        """Test with very large weights."""
        data = [(("A", "B"), 1e6), (("B", "C"), 1e6)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should handle large weights
        assert model.is_fitted
        assert not np.any(np.isnan(model.scores))
        assert not np.any(np.isinf(model.scores))

    def test_very_small_weights(self):
        """Test with very small weights."""
        data = [(("A", "B"), 1e-6), (("B", "C"), 1e-6)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should handle small weights
        assert model.is_fitted
        assert not np.any(np.isnan(model.scores))

    def test_mixed_weight_scales(self):
        """Test with mixed weight scales."""
        data = [
            (("A", "B"), 1),
            (("B", "C"), 1000),
            (("C", "D"), 0.001),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should handle mixed scales
        assert model.is_fitted
        assert all(score > 0 for score in model.scores)

    def test_extreme_score_ratios(self):
        """Test when one node is extremely dominant."""
        data = [
            (("A", "B"), 100),
            (("A", "C"), 100),
            (("B", "C"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        score_a = model.get_score("A")
        score_b = model.get_score("B")
        score_c = model.get_score("C")

        # A should be much higher than B and C
        assert score_a > 10 * score_b
        assert score_a > 10 * score_c

    def test_convergence_with_noise(self):
        """Test convergence with noisy data."""
        # Add random noise to consistent pattern
        np.random.seed(42)
        base_data = [
            (("A", "B", "C"), 10),
            (("A", "C", "B"), 8),
            (("B", "C", "A"), 2),
        ]

        # Add random contradictory data
        noise = generate_synthetic_rankings(N=3, M=20, K_min=2, K_max=3, seed=99)
        data = base_data + noise

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should still converge
        assert stats["iterations"] < model.max_iterations

    def test_very_tight_epsilon(self):
        """Test with very tight convergence criterion."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model = PlackettLuceModel(model_type="full", epsilon=1e-10)
        model.fit(data, verbose=False)

        # Might take more iterations but should converge
        assert model.is_fitted

    def test_floating_point_precision(self):
        """Test that scores maintain precision."""
        data = [(("A", "B", "C"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Scores should sum to reasonable value
        geometric_mean = np.exp(np.mean(np.log(model.scores)))
        assert np.isclose(geometric_mean, 1.0, atol=1e-8)


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    def test_maximum_nodes(self):
        """Test with maximum practical number of nodes."""
        N = 500
        data = generate_synthetic_rankings(N=N, M=1000, K_min=2, K_max=10, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert len(model.scores) == N
        assert stats["time"] < 120  # Should complete in reasonable time

    def test_maximum_comparison_size(self):
        """Test with very large comparison size."""
        nodes = [f"N{i}" for i in range(100)]
        data = [(tuple(nodes), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert len(model.scores) == 100

    def test_minimum_data_for_ranking(self):
        """Test minimum data needed for meaningful ranking."""
        # Just enough to establish order
        data = [(("A", "B"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 2

    def test_maximum_iterations_reached(self):
        """Test behavior when max iterations reached."""
        data = generate_synthetic_rankings(N=20, M=100, seed=42)

        model = PlackettLuceModel(model_type="full", epsilon=1e-10, max_iterations=10)
        stats = model.fit(data, verbose=False)

        # Should stop at max iterations
        assert stats["iterations"] == 10
        assert model.is_fitted  # Should still be usable

    def test_epsilon_zero(self):
        """Test with epsilon of zero (always iterate to max)."""
        data = [(("A", "B"), 1)]

        model = PlackettLuceModel(model_type="full", epsilon=0.0, max_iterations=5)
        stats = model.fit(data, verbose=False)

        # Should always hit max iterations
        assert stats["iterations"] == 5


class TestSpecialCharacters:
    """Test handling of special characters in node IDs."""

    def test_unicode_node_names(self):
        """Test with unicode node names."""
        data = [
            (("François", "José", "Müller"), 1),
            (("José", "François"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 3

    def test_special_character_names(self):
        """Test with special characters in names."""
        data = [
            (("Team-A", "Team_B", "Team.C"), 1),
            (("Team@1", "Team#2"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert model.is_fitted

    def test_numeric_node_ids(self):
        """Test with numeric node IDs."""
        data = [
            ((1, 2, 3), 1),
            ((2, 3, 1), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 3

    def test_mixed_type_nodes(self):
        """Test with mixed type node IDs."""
        data = [
            ((1, "B", 3.0), 1),
            (("B", 1), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should handle mixed types
        assert model.is_fitted

    def test_empty_string_node(self):
        """Test with empty string as node ID."""
        data = [(("A", "", "C"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should work (empty string is valid)
        assert model.is_fitted

    def test_whitespace_node_names(self):
        """Test with whitespace in node names."""
        data = [
            (("Team A", "Team B", "Team C"), 1),
            (("Team B", "Team A"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 3


class TestDataConsistency:
    """Test data consistency and validation."""

    def test_data_immutability(self):
        """Test that fitting doesn't modify input data."""
        data = [
            (("A", "B", "C"), 1),
            (("B", "C"), 2),
        ]
        data_copy = [item for item in data]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Data should be unchanged
        assert data == data_copy

    def test_repeated_fitting_consistency(self):
        """Test that repeated fitting gives same results."""
        data = generate_synthetic_rankings(N=15, M=100, seed=42)

        model = PlackettLuceModel(model_type="full", epsilon=1e-8)

        # Fit multiple times
        model.fit(data, verbose=False)
        ranking1 = model.get_ranking()

        model.fit(data, verbose=False)
        ranking2 = model.get_ranking()

        model.fit(data, verbose=False)
        ranking3 = model.get_ranking()

        # All rankings should be identical
        assert ranking1 == ranking2 == ranking3

    def test_score_positivity(self):
        """Test that all scores are positive."""
        data = generate_synthetic_rankings(N=20, M=100, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert all(score > 0 for score in model.scores)

    def test_score_normalization_preserved(self):
        """Test that score normalization is preserved."""
        data = generate_synthetic_rankings(N=15, M=100, seed=42)

        for model_type in ["full", "position1"]:
            model = PlackettLuceModel(model_type=model_type)
            model.fit(data, verbose=False)

            geometric_mean = np.exp(np.mean(np.log(model.scores)))
            assert np.isclose(geometric_mean, 1.0, atol=1e-6)

    def test_log_likelihood_negativity(self):
        """Test that log-likelihood is always negative."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ll = model.log_likelihood(data)
        assert ll < 0

    def test_probability_bounds(self):
        """Test that predicted probabilities are in [0,1]."""
        data = [
            (("A", "B", "C"), 1),
            (("B", "A", "C"), 1),
        ]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Test various predictions
        for ranking in [("A", "B", "C"), ("B", "A", "C"), ("C", "A", "B")]:
            prob = model.predict_probability(ranking)
            assert 0 <= prob <= 1

    def test_probability_sum(self):
        """Test that probabilities over all rankings sum to reasonable value."""
        data = [(("A", "B", "C"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Get probabilities for all 6 possible rankings
        from itertools import permutations

        nodes = ["A", "B", "C"]
        total_prob = 0

        for perm in permutations(nodes):
            prob = model.predict_probability(perm)
            total_prob += prob

        # Should sum to approximately 1
        assert np.isclose(total_prob, 1.0, atol=1e-6)


class TestProjectedEdgeCases:
    """Test edge cases specific to projected models."""

    def test_projected_empty_data(self):
        """Test projected model with empty data."""
        model = ProjectedPlackettLuce(model_type="full")

        with pytest.raises(ValueError):
            model.fit([], verbose=False)

    def test_projected_single_pair(self):
        """Test projected model reduces to identity for pairs."""
        data = [(("A", "B"), 1)]

        model_pl = PlackettLuceModel(model_type="full")
        model_proj = ProjectedPlackettLuce(model_type="full")

        model_pl.fit(data, verbose=False)
        model_proj.fit(data, verbose=False)

        # Should be identical for pairwise data
        assert np.allclose(model_pl.scores, model_proj.scores, atol=1e-4)

    def test_projected_extreme_comparison_size(self):
        """Test projected model with extreme comparison size."""
        nodes = [f"N{i}" for i in range(20)]
        data = [(tuple(nodes), 1)]

        model = ProjectedPlackettLuce(model_type="full")
        model.fit(data, verbose=False)

        # Should create many pairwise comparisons
        assert model.is_fitted

    def test_projected_position1_single_winner(self):
        """Test position-1 projection with consistent winner."""
        data = [
            (("A", "B", "C"), 5),
            (("A", "D", "E"), 5),
            (("A", "F", "G"), 5),
        ]

        model = ProjectedPlackettLuce(model_type="position1")
        model.fit(data, verbose=False)

        # A should be top ranked
        ranking = model.get_ranking()
        assert ranking[0][0] == "A"


class TestConcurrentUsage:
    """Test edge cases in concurrent or multiple model usage."""

    def test_multiple_models_same_data(self):
        """Test multiple models on same data don't interfere."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model1 = PlackettLuceModel(model_type="full")
        model2 = PlackettLuceModel(model_type="full")

        model1.fit(data, verbose=False)
        model2.fit(data, verbose=False)

        # Should produce identical results
        assert np.allclose(model1.scores, model2.scores)

    def test_different_model_types_same_data(self):
        """Test different model types on same data."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model_full = PlackettLuceModel(model_type="full")
        model_pos1 = PlackettLuceModel(model_type="position1")

        model_full.fit(data, verbose=False)
        model_pos1.fit(data, verbose=False)

        # Results should differ
        assert not np.allclose(model_full.scores, model_pos1.scores)

    def test_refit_different_data(self):
        """Test refitting same model with different data."""
        data1 = [(("A", "B"), 1)]
        data2 = [(("B", "A"), 1)]

        model = PlackettLuceModel(model_type="full")

        model.fit(data1, verbose=False)
        ranking1 = model.get_ranking()

        model.fit(data2, verbose=False)
        ranking2 = model.get_ranking()

        # Rankings should be reversed
        assert ranking1[0][0] != ranking2[0][0]


class TestRobustness:
    """Test model robustness to various conditions."""

    def test_robustness_to_initialization(self):
        """Test that results don't depend on initialization."""
        data = generate_synthetic_rankings(N=15, M=100, seed=42)

        rankings = []
        for seed in range(5):
            np.random.seed(seed)
            model = PlackettLuceModel(model_type="full")
            model.fit(data, verbose=False)
            rankings.append([node for node, _ in model.get_ranking()])

        # All should produce similar rankings
        from plackett_luce.utils import ranking_similarity

        similarities = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                sim = ranking_similarity(rankings[i], rankings[j], method="kendall")
                similarities.append(sim)

        # Should be very similar (close to 1)
        assert np.mean(similarities) > 0.9

    def test_robustness_to_data_order(self):
        """Test that data order doesn't affect results."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        # Fit with original order
        model1 = PlackettLuceModel(model_type="full")
        model1.fit(data, verbose=False)

        # Fit with shuffled order
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)
        model2 = PlackettLuceModel(model_type="full")
        model2.fit(shuffled_data, verbose=False)

        # Should produce identical results
        assert np.allclose(model1.scores, model2.scores, atol=1e-6)

    def test_robustness_to_epsilon(self):
        """Test that reasonable epsilon values give similar results."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        rankings = []
        for epsilon in [1e-4, 1e-5, 1e-6, 1e-7]:
            model = PlackettLuceModel(model_type="full", epsilon=epsilon)
            model.fit(data, verbose=False)
            rankings.append([node for node, _ in model.get_ranking()])

        # All should produce identical or very similar rankings
        assert all(rankings[0] == r for r in rankings)
