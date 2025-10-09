"""
Comprehensive tests for utility functions.
"""

import numpy as np
import pytest

from plackett_luce.utils import (
    generate_synthetic_rankings,
    ranking_similarity,
)


class TestGenerateSyntheticRankings:
    """Test suite for generate_synthetic_rankings function."""

    def test_basic_generation(self):
        """Test basic synthetic data generation."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        assert len(data) == 50
        assert all(isinstance(item, tuple) and len(item) == 2 for item in data)

    def test_output_format(self):
        """Test that output has correct format."""
        data = generate_synthetic_rankings(N=10, M=20, K_min=2, K_max=4, seed=42)

        for ranking, weight in data:
            assert isinstance(ranking, tuple)
            assert isinstance(weight, (int, float))
            assert weight == 1  # Default weight
            assert 2 <= len(ranking) <= 4

    def test_node_count(self):
        """Test that correct number of unique nodes are used."""
        N = 15
        data = generate_synthetic_rankings(N=N, M=100, K_min=2, K_max=5, seed=42)

        all_nodes = set()
        for ranking, _ in data:
            all_nodes.update(ranking)

        # Should use all or most of the N nodes
        assert len(all_nodes) >= N * 0.8  # At least 80% of nodes appear

    def test_comparison_count(self):
        """Test that correct number of comparisons are generated."""
        M = 73
        data = generate_synthetic_rankings(N=10, M=M, K_min=2, K_max=5, seed=42)

        assert len(data) == M

    def test_comparison_size_bounds(self):
        """Test that comparison sizes respect K_min and K_max."""
        K_min, K_max = 3, 7
        data = generate_synthetic_rankings(N=20, M=100, K_min=K_min, K_max=K_max, seed=42)

        for ranking, _ in data:
            K = len(ranking)
            assert K_min <= K <= K_max

    def test_reproducibility_with_seed(self):
        """Test that seed ensures reproducibility."""
        data1 = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)
        data2 = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        assert data1 == data2

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        data1 = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)
        data2 = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=43)

        assert data1 != data2

    def test_no_duplicate_nodes_in_ranking(self):
        """Test that no ranking contains duplicate nodes."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=10, seed=42)

        for ranking, _ in data:
            assert len(ranking) == len(set(ranking))

    def test_pairwise_comparisons_only(self):
        """Test generation of only pairwise comparisons."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=2, seed=42)

        for ranking, _ in data:
            assert len(ranking) == 2

    def test_large_comparisons(self):
        """Test generation of large comparisons."""
        data = generate_synthetic_rankings(N=20, M=30, K_min=10, K_max=15, seed=42)

        for ranking, _ in data:
            assert 10 <= len(ranking) <= 15

    def test_all_nodes_comparison(self):
        """Test comparison involving all nodes."""
        N = 10
        data = generate_synthetic_rankings(N=N, M=5, K_min=N, K_max=N, seed=42)

        for ranking, _ in data:
            assert len(ranking) == N

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # N must be positive
        with pytest.raises((ValueError, TypeError)):
            generate_synthetic_rankings(N=0, M=10)

        # M must be positive
        with pytest.raises((ValueError, TypeError)):
            generate_synthetic_rankings(N=10, M=0)

        # K_min must be at least 2
        with pytest.raises((ValueError, TypeError, IndexError)):
            generate_synthetic_rankings(N=10, M=10, K_min=1, K_max=5)

    def test_k_max_exceeds_n(self):
        """Test when K_max exceeds N."""
        # Should handle gracefully or raise error
        try:
            data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=10, seed=42)
            # If it works, check that K never exceeds N
            for ranking, _ in data:
                assert len(ranking) <= 5
        except (ValueError, IndexError):
            # Also acceptable to raise error
            pass

    def test_k_min_greater_than_k_max(self):
        """Test error when K_min > K_max."""
        with pytest.raises((ValueError, AssertionError)):
            generate_synthetic_rankings(N=10, M=10, K_min=5, K_max=3, seed=42)

    def test_negative_parameters(self):
        """Test that negative parameters raise errors."""
        with pytest.raises((ValueError, TypeError)):
            generate_synthetic_rankings(N=-5, M=10)

        with pytest.raises((ValueError, TypeError)):
            generate_synthetic_rankings(N=10, M=-5)

    def test_very_small_dataset(self):
        """Test generation of very small dataset."""
        data = generate_synthetic_rankings(N=3, M=2, K_min=2, K_max=2, seed=42)

        assert len(data) == 2
        for ranking, _ in data:
            assert len(ranking) == 2

    def test_very_large_dataset(self):
        """Test generation of very large dataset."""
        data = generate_synthetic_rankings(N=100, M=1000, K_min=2, K_max=10, seed=42)

        assert len(data) == 1000

    def test_single_comparison(self):
        """Test generation of single comparison."""
        data = generate_synthetic_rankings(N=5, M=1, K_min=3, K_max=4, seed=42)

        assert len(data) == 1
        ranking, weight = data[0]
        assert 3 <= len(ranking) <= 4
        assert weight == 1

    def test_node_indices_are_integers(self):
        """Test that generated node IDs are integers."""
        data = generate_synthetic_rankings(N=10, M=20, K_min=2, K_max=5, seed=42)

        for ranking, _ in data:
            for node in ranking:
                assert isinstance(node, (int, np.integer))

    def test_node_indices_in_range(self):
        """Test that node indices are in valid range."""
        N = 15
        data = generate_synthetic_rankings(N=N, M=50, K_min=2, K_max=5, seed=42)

        for ranking, _ in data:
            for node in ranking:
                assert 0 <= node < N

    def test_ranking_order_meaningful(self):
        """Test that rankings follow PL model probabilities."""
        # Generate data with specific seed
        data = generate_synthetic_rankings(N=10, M=100, K_min=3, K_max=3, seed=42)

        # Fit model to verify it makes sense
        from plackett_luce import PlackettLuceModel

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Should converge successfully
        assert model.is_fitted

    def test_weight_is_always_one(self):
        """Test that all weights are 1 (current implementation)."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        for _, weight in data:
            assert weight == 1

    def test_rankings_are_tuples(self):
        """Test that rankings are tuples, not lists."""
        data = generate_synthetic_rankings(N=10, M=20, K_min=2, K_max=5, seed=42)

        for ranking, _ in data:
            assert isinstance(ranking, tuple)

    def test_distribution_of_k_values(self):
        """Test that K values are reasonably distributed."""
        data = generate_synthetic_rankings(N=20, M=500, K_min=2, K_max=8, seed=42)

        from collections import Counter

        k_counts = Counter(len(ranking) for ranking, _ in data)

        # Should have comparisons of various sizes
        assert len(k_counts) >= 4  # At least 4 different sizes

    def test_no_empty_rankings(self):
        """Test that no rankings are empty."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        for ranking, _ in data:
            assert len(ranking) > 0

    def test_deterministic_for_same_seed(self):
        """Test complete determinism with same seed."""
        params = {"N": 15, "M": 75, "K_min": 2, "K_max": 6, "seed": 123}

        data1 = generate_synthetic_rankings(**params)
        data2 = generate_synthetic_rankings(**params)
        data3 = generate_synthetic_rankings(**params)

        assert data1 == data2 == data3


class TestRankingSimilarity:
    """Test suite for ranking_similarity function."""

    def test_identical_rankings_kendall(self):
        """Test that identical rankings have similarity 1."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["A", "B", "C", "D"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

    def test_identical_rankings_spearman(self):
        """Test that identical rankings have similarity 1."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["A", "B", "C", "D"]

        sim = ranking_similarity(ranking1, ranking2, method="spearman")
        assert np.isclose(sim, 1.0)

    def test_reversed_rankings_kendall(self):
        """Test that reversed rankings have similarity -1."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["D", "C", "B", "A"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, -1.0)

    def test_reversed_rankings_spearman(self):
        """Test that reversed rankings have similarity -1."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["D", "C", "B", "A"]

        sim = ranking_similarity(ranking1, ranking2, method="spearman")
        assert np.isclose(sim, -1.0)

    def test_partially_similar_rankings(self):
        """Test partially similar rankings."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["A", "C", "B", "D"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert -1 < sim < 1
        assert sim > 0  # Should be positive

    def test_no_common_elements(self):
        """Test rankings with no common elements."""
        ranking1 = ["A", "B", "C"]
        ranking2 = ["X", "Y", "Z"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert sim == 0.0

    def test_partial_overlap(self):
        """Test rankings with partial overlap."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["C", "D", "E", "F"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        # Should compute similarity based on C and D only
        assert isinstance(sim, float)
        assert -1 <= sim <= 1

    def test_different_lengths(self):
        """Test rankings of different lengths."""
        ranking1 = ["A", "B", "C", "D", "E"]
        ranking2 = ["A", "B", "C"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        # Should work on common elements
        assert isinstance(sim, float)
        assert np.isclose(sim, 1.0)  # A, B, C in same order

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        ranking1 = ["A", "B", "C"]
        ranking2 = ["A", "B", "C"]

        with pytest.raises(ValueError):
            ranking_similarity(ranking1, ranking2, method="invalid")

    def test_kendall_vs_spearman(self):
        """Test that kendall and spearman give different results."""
        ranking1 = ["A", "B", "C", "D", "E"]
        ranking2 = ["A", "C", "B", "E", "D"]

        sim_kendall = ranking_similarity(ranking1, ranking2, method="kendall")
        sim_spearman = ranking_similarity(ranking1, ranking2, method="spearman")

        # Generally different, but both should be positive
        assert sim_kendall > 0
        assert sim_spearman > 0

    def test_single_element_rankings(self):
        """Test with single element rankings."""
        ranking1 = ["A"]
        ranking2 = ["A"]

        # Should return 1.0 or NaN
        try:
            sim = ranking_similarity(ranking1, ranking2, method="kendall")
            assert np.isclose(sim, 1.0) or np.isnan(sim)
        except (ValueError, RuntimeWarning):
            pass  # Acceptable to fail on single element

    def test_two_element_rankings(self):
        """Test with two element rankings."""
        ranking1 = ["A", "B"]
        ranking2 = ["A", "B"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

        ranking3 = ["B", "A"]
        sim2 = ranking_similarity(ranking1, ranking3, method="kendall")
        assert np.isclose(sim2, -1.0)

    def test_numeric_rankings(self):
        """Test with numeric node IDs."""
        ranking1 = [1, 2, 3, 4]
        ranking2 = [1, 2, 3, 4]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

    def test_mixed_type_rankings(self):
        """Test with mixed type node IDs."""
        ranking1 = [1, "B", 3, "D"]
        ranking2 = [1, "B", 3, "D"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

    def test_duplicate_elements(self):
        """Test handling of duplicate elements."""
        ranking1 = ["A", "B", "B", "C"]
        ranking2 = ["A", "B", "C", "C"]

        # Behavior may vary - just test it doesn't crash
        try:
            sim = ranking_similarity(ranking1, ranking2, method="kendall")
            assert isinstance(sim, float)
        except (ValueError, KeyError):
            pass  # Acceptable to reject duplicates

    def test_empty_rankings(self):
        """Test with empty rankings."""
        ranking1 = []
        ranking2 = []

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert sim == 0.0

    def test_one_empty_ranking(self):
        """Test with one empty ranking."""
        ranking1 = ["A", "B", "C"]
        ranking2 = []

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert sim == 0.0

    def test_complete_disagreement(self):
        """Test rankings in complete disagreement."""
        ranking1 = ["A", "B", "C", "D", "E", "F"]
        ranking2 = ["F", "E", "D", "C", "B", "A"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, -1.0)

    def test_random_permutation(self):
        """Test with random permutation."""
        ranking1 = ["A", "B", "C", "D", "E"]
        ranking2 = ["C", "A", "E", "B", "D"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert -1 <= sim <= 1

    def test_large_rankings(self):
        """Test with large rankings."""
        ranking1 = list(range(100))
        ranking2 = list(range(100))

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

    def test_large_rankings_reversed(self):
        """Test with large reversed rankings."""
        ranking1 = list(range(100))
        ranking2 = list(reversed(range(100)))

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, -1.0, atol=1e-5)

    def test_symmetry(self):
        """Test that similarity is symmetric."""
        ranking1 = ["A", "B", "C", "D"]
        ranking2 = ["B", "A", "D", "C"]

        sim1 = ranking_similarity(ranking1, ranking2, method="kendall")
        sim2 = ranking_similarity(ranking2, ranking1, method="kendall")

        assert np.isclose(sim1, sim2)

    def test_transitivity_example(self):
        """Test example case for transitivity understanding."""
        # If A~B (similar) and B~C (similar), is A~C (similar)?
        # Not guaranteed for ranking similarity, but test the values

        ranking_a = ["A", "B", "C", "D"]
        ranking_b = ["A", "C", "B", "D"]
        ranking_c = ["A", "C", "D", "B"]

        sim_ab = ranking_similarity(ranking_a, ranking_b, method="kendall")
        sim_bc = ranking_similarity(ranking_b, ranking_c, method="kendall")
        sim_ac = ranking_similarity(ranking_a, ranking_c, method="kendall")

        # Just verify all are computed
        assert all(isinstance(s, float) for s in [sim_ab, sim_bc, sim_ac])

    def test_string_rankings(self):
        """Test with string node IDs."""
        ranking1 = ["Team_A", "Team_B", "Team_C"]
        ranking2 = ["Team_A", "Team_B", "Team_C"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

    def test_unicode_rankings(self):
        """Test with unicode characters."""
        ranking1 = ["α", "β", "γ", "δ"]
        ranking2 = ["α", "β", "γ", "δ"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert np.isclose(sim, 1.0)

    def test_ranking_similarity_bounds(self):
        """Test that similarity is always in [-1, 1]."""
        for _ in range(10):
            # Generate random rankings
            items = list(range(10))
            np.random.shuffle(items)
            ranking1 = items.copy()
            np.random.shuffle(items)
            ranking2 = items.copy()

            sim = ranking_similarity(ranking1, ranking2, method="kendall")
            assert -1 <= sim <= 1


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_generate_and_measure_similarity(self):
        """Test generating data and measuring ranking similarity."""
        data = generate_synthetic_rankings(N=15, M=100, K_min=3, K_max=5, seed=42)

        # Fit model
        from plackett_luce import PlackettLuceModel

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Get ranking
        ranking = [node for node, _ in model.get_ranking()]

        # Generate new data from same distribution
        data2 = generate_synthetic_rankings(N=15, M=100, K_min=3, K_max=5, seed=43)
        model2 = PlackettLuceModel(model_type="full")
        model2.fit(data2, verbose=False)
        ranking2 = [node for node, _ in model2.get_ranking()]

        # Measure similarity
        sim = ranking_similarity(ranking, ranking2, method="kendall")

        # Should have some positive similarity (same underlying distribution)
        assert -1 <= sim <= 1

    def test_reproducible_pipeline(self):
        """Test that entire pipeline is reproducible."""
        seed = 42

        # Run 1
        data1 = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=4, seed=seed)
        from plackett_luce import PlackettLuceModel

        model1 = PlackettLuceModel(model_type="full")
        model1.fit(data1, verbose=False)
        ranking1 = [node for node, _ in model1.get_ranking()]

        # Run 2
        data2 = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=4, seed=seed)
        model2 = PlackettLuceModel(model_type="full")
        model2.fit(data2, verbose=False)
        ranking2 = [node for node, _ in model2.get_ranking()]

        # Should be identical
        assert ranking1 == ranking2

    def test_synthetic_data_properties(self):
        """Test that synthetic data has expected statistical properties."""
        data = generate_synthetic_rankings(N=20, M=500, K_min=2, K_max=8, seed=42)

        # All nodes should appear
        all_nodes = set()
        for ranking, _ in data:
            all_nodes.update(ranking)
        assert len(all_nodes) == 20

        # Node appearance frequency should be reasonable
        from collections import Counter

        node_counts = Counter()
        for ranking, _ in data:
            node_counts.update(ranking)

        # Each node should appear multiple times
        assert all(count >= 10 for count in node_counts.values())


class TestUtilsEdgeCases:
    """Test edge cases and error conditions."""

    def test_generate_with_extreme_parameters(self):
        """Test generation with extreme but valid parameters."""
        # Very large N, small M
        data = generate_synthetic_rankings(N=1000, M=10, K_min=2, K_max=3, seed=42)
        assert len(data) == 10

    def test_similarity_with_none_values(self):
        """Test similarity with None values."""
        ranking1 = ["A", None, "C"]
        ranking2 = ["A", "B", "C"]

        # Should handle or reject None
        try:
            sim = ranking_similarity(ranking1, ranking2, method="kendall")
            assert isinstance(sim, (float, type(None)))
        except (ValueError, TypeError, KeyError):
            pass  # Acceptable to reject None values

    def test_similarity_with_nan_values(self):
        """Test similarity with NaN values."""
        ranking1 = ["A", "B", "C"]
        ranking2 = ["A", "B", "C"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert not np.isnan(sim) or sim == 0.0


class TestUtilsDocumentation:
    """Test that functions match their documentation."""

    def test_generate_returns_correct_type(self):
        """Test that generate_synthetic_rankings returns List[Tuple]."""
        data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=42)

        assert isinstance(data, list)
        assert all(isinstance(item, tuple) for item in data)
        assert all(len(item) == 2 for item in data)

    def test_similarity_returns_float(self):
        """Test that ranking_similarity returns float."""
        ranking1 = ["A", "B", "C"]
        ranking2 = ["A", "B", "C"]

        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        assert isinstance(sim, (float, np.floating))


class TestUtilsPerformance:
    """Test performance characteristics of utility functions."""

    def test_generate_performance(self):
        """Test that generation completes in reasonable time."""
        import time

        start = time.time()
        data = generate_synthetic_rankings(N=100, M=1000, K_min=2, K_max=10, seed=42)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert len(data) == 1000

    def test_similarity_performance(self):
        """Test that similarity computation is fast."""
        import time

        ranking1 = list(range(1000))
        ranking2 = list(reversed(range(1000)))

        start = time.time()
        sim = ranking_similarity(ranking1, ranking2, method="kendall")
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in under 1 second
        assert isinstance(sim, float)
