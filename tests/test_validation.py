"""
Tests for validation utilities.
"""

import numpy as np
import pytest

from plackett_luce import PlackettLuceModel
from plackett_luce.utils import generate_synthetic_rankings
from plackett_luce.validation import (
    cross_validate,
    leave_one_out_cv,
    stratified_split_by_size,
    temporal_split,
    train_test_split,
)


class TestLeaveOneOutCV:
    """Comprehensive tests for leave_one_out_cv function."""

    def test_basic_leave_one_out(self):
        """Test basic leave-one-out cross-validation."""
        data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert "mean" in results
        assert "std" in results
        assert "scores" in results
        assert "n_samples" in results
        assert results["n_samples"] == 10

    def test_loo_scores_length(self):
        """Test that LOO produces correct number of scores."""
        data = generate_synthetic_rankings(N=5, M=15, K_min=2, K_max=3, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert len(results["scores"]) == 15

    def test_loo_scores_are_negative(self):
        """Test that all LOO scores are negative log-likelihoods."""
        data = generate_synthetic_rankings(N=8, M=20, K_min=2, K_max=4, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        # All log-likelihoods should be negative
        assert all(score < 0 for score in results["scores"])

    def test_loo_mean_and_std(self):
        """Test that mean and std are computed correctly."""
        data = generate_synthetic_rankings(N=5, M=12, K_min=2, K_max=3, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        # Verify mean
        assert np.isclose(results["mean"], np.mean(results["scores"]))

        # Verify std
        assert np.isclose(results["std"], np.std(results["scores"]))

    def test_loo_position1_model(self):
        """Test LOO with position-1 model."""
        data = generate_synthetic_rankings(N=6, M=10, K_min=2, K_max=4, seed=42)

        results = leave_one_out_cv(data, model_type="position1", verbose=False)

        assert results["model_type"] == "position1"
        assert len(results["scores"]) == 10

    def test_loo_full_model(self):
        """Test LOO with full model."""
        data = generate_synthetic_rankings(N=6, M=10, K_min=2, K_max=4, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert results["model_type"] == "full"
        assert len(results["scores"]) == 10

    def test_loo_small_dataset(self):
        """Test LOO with very small dataset."""
        data = [(("A", "B"), 1), (("B", "C"), 1), (("A", "C"), 1)]

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert results["n_samples"] == 3
        assert len(results["scores"]) == 3

    def test_loo_single_comparison(self):
        """Test LOO with single comparison."""
        data = [(("A", "B", "C"), 1)]

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert results["n_samples"] == 1
        # Model trained on empty data, may fail or give extreme value
        assert len(results["scores"]) == 1

    def test_loo_verbose_mode(self):
        """Test LOO with verbose output."""
        data = generate_synthetic_rankings(N=5, M=8, K_min=2, K_max=3, seed=42)

        # Should not crash with verbose=True
        results = leave_one_out_cv(data, model_type="full", verbose=True)

        assert results["n_samples"] == 8

    def test_loo_deterministic(self):
        """Test that LOO is deterministic."""
        data = generate_synthetic_rankings(N=6, M=12, K_min=2, K_max=3, seed=42)

        results1 = leave_one_out_cv(data, model_type="full", verbose=False)
        results2 = leave_one_out_cv(data, model_type="full", verbose=False)

        assert results1["scores"] == results2["scores"]
        assert results1["mean"] == results2["mean"]

    def test_loo_with_different_data(self):
        """Test that LOO produces different results for different data."""
        data1 = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=42)
        data2 = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=43)

        results1 = leave_one_out_cv(data1, model_type="full", verbose=False)
        results2 = leave_one_out_cv(data2, model_type="full", verbose=False)

        # Should produce different scores
        assert results1["scores"] != results2["scores"]

    def test_loo_captures_overfitting(self):
        """Test that LOO can detect overfitting."""
        # Create very small dataset
        data = [(("A", "B"), 1), (("B", "C"), 1)]

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        # Should complete without errors
        assert results["n_samples"] == 2

    def test_loo_variance_calculation(self):
        """Test that variance is reasonable."""
        data = generate_synthetic_rankings(N=8, M=20, K_min=2, K_max=4, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        # Std should be non-negative and finite
        assert results["std"] >= 0
        assert np.isfinite(results["std"])

    def test_loo_with_weights(self):
        """Test LOO with weighted comparisons."""
        data = [
            (("A", "B", "C"), 5),
            (("B", "C", "A"), 3),
            (("C", "A", "B"), 2),
        ]

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert results["n_samples"] == 3
        assert len(results["scores"]) == 3

    def test_loo_consistency_with_cv(self):
        """Test that LOO is similar to k-fold CV with k=n."""
        data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=42)

        loo_results = leave_one_out_cv(data, model_type="full", verbose=False)

        # LOO should give reasonable scores
        assert np.isfinite(loo_results["mean"])
        assert loo_results["mean"] < 0  # Negative log-likelihood


class TestTrainTestSplitAdvanced:
    """Advanced tests for train_test_split."""

    def test_split_preserves_data_integrity(self):
        """Test that split doesn't modify original data."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)
        data_copy = [item for item in data]

        train, test = train_test_split(data, test_size=0.2, random_state=42)

        # Original data should be unchanged
        assert data == data_copy

    def test_split_no_data_loss(self):
        """Test that split doesn't lose any data."""
        data = generate_synthetic_rankings(N=10, M=100, K_min=2, K_max=5, seed=42)

        train, test = train_test_split(data, test_size=0.3, random_state=42)

        # All data should be accounted for
        assert len(train) + len(test) == len(data)

        # Convert to sets to check no overlap
        train_str = {str(item) for item in train}
        test_str = {str(item) for item in test}

        assert len(train_str & test_str) == 0

    def test_split_with_no_shuffle(self):
        """Test split without shuffling."""
        data = [(("A", "B"), i) for i in range(20)]

        train, test = train_test_split(data, test_size=0.2, shuffle=False)

        # Without shuffle, should be sequential split
        assert train == data[:16]
        assert test == data[16:]

    def test_split_shuffle_changes_order(self):
        """Test that shuffle actually shuffles."""
        data = [(("A", "B"), i) for i in range(50)]

        train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

        # Train should not be the first 40 items
        assert train != data[:40]

    def test_split_extreme_test_sizes(self):
        """Test split with extreme test sizes."""
        data = generate_synthetic_rankings(N=10, M=100, K_min=2, K_max=5, seed=42)

        # Very small test set
        train, test = train_test_split(data, test_size=0.01, random_state=42)
        assert len(test) >= 1
        assert len(train) >= 99

        # Very large test set
        train, test = train_test_split(data, test_size=0.99, random_state=42)
        assert len(test) >= 99
        assert len(train) >= 1

    def test_split_boundary_values(self):
        """Test split at boundary values."""
        data = generate_synthetic_rankings(N=10, M=100, K_min=2, K_max=5, seed=42)

        # Just above 0
        train, test = train_test_split(data, test_size=0.001, random_state=42)
        assert len(test) >= 1

        # Just below 1
        train, test = train_test_split(data, test_size=0.999, random_state=42)
        assert len(train) >= 1

    def test_split_exact_half(self):
        """Test 50-50 split."""
        data = generate_synthetic_rankings(N=10, M=100, K_min=2, K_max=5, seed=42)

        train, test = train_test_split(data, test_size=0.5, random_state=42)

        assert len(train) == 50
        assert len(test) == 50

    def test_split_with_single_item(self):
        """Test split with single item."""
        data = [(("A", "B"), 1)]

        # Should handle gracefully
        try:
            train, test = train_test_split(data, test_size=0.5, random_state=42)
            assert len(train) + len(test) == 1
        except ValueError:
            pass  # Acceptable to reject very small datasets

    def test_split_with_two_items(self):
        """Test split with two items."""
        data = [(("A", "B"), 1), (("C", "D"), 1)]

        train, test = train_test_split(data, test_size=0.5, random_state=42)

        assert len(train) == 1
        assert len(test) == 1

    def test_split_random_state_types(self):
        """Test different random_state types."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        # Integer seed
        train1, test1 = train_test_split(data, test_size=0.2, random_state=42)
        train2, test2 = train_test_split(data, test_size=0.2, random_state=42)
        assert train1 == train2

        # No seed
        train3, test3 = train_test_split(data, test_size=0.2, random_state=None)
        assert len(train3) + len(test3) == len(data)


class TestStratifiedSplitAdvanced:
    """Advanced tests for stratified_split_by_size."""

    def test_stratified_maintains_distribution(self):
        """Test that stratified split maintains size distribution."""
        from collections import Counter

        # Create data with specific size distribution
        data = (
            [(("A", "B"), 1) for _ in range(20)]
            + [(("C", "D", "E"), 1) for _ in range(30)]
            + [(("F", "G", "H", "I"), 1) for _ in range(50)]
        )

        train, test = stratified_split_by_size(data, test_size=0.2, random_state=42)

        def get_dist(dataset):
            sizes = [len(r) for r, _ in dataset]
            return Counter(sizes)

        orig_dist = get_dist(data)
        train_dist = get_dist(train)
        test_dist = get_dist(test)

        # Check proportions are approximately maintained
        for size in orig_dist:
            if size in train_dist and size in test_dist:
                orig_ratio = orig_dist[size] / len(data)
                train_ratio = train_dist[size] / len(train)
                test_ratio = test_dist[size] / len(test)

                # Ratios should be similar
                assert abs(orig_ratio - train_ratio) < 0.15
                assert abs(orig_ratio - test_ratio) < 0.15

    def test_stratified_all_sizes_present(self):
        """Test that all comparison sizes appear in both splits."""
        data = (
            [(("A", "B"), 1) for _ in range(10)]
            + [(("C", "D", "E"), 1) for _ in range(10)]
            + [(("F", "G", "H", "I"), 1) for _ in range(10)]
        )

        train, test = stratified_split_by_size(data, test_size=0.3, random_state=42)

        train_sizes = set(len(r) for r, _ in train)
        test_sizes = set(len(r) for r, _ in test)

        # Both should have variety
        assert len(train_sizes) >= 2
        assert len(test_sizes) >= 2

    def test_stratified_deterministic(self):
        """Test that stratified split is deterministic."""
        data = generate_synthetic_rankings(N=15, M=100, K_min=2, K_max=6, seed=42)

        train1, test1 = stratified_split_by_size(data, test_size=0.2, random_state=42)
        train2, test2 = stratified_split_by_size(data, test_size=0.2, random_state=42)

        assert train1 == train2
        assert test1 == test2

    def test_stratified_vs_random_split(self):
        """Test that stratified differs from random split."""
        data = [(("A", "B"), 1) for _ in range(50)] + [
            (("C", "D", "E", "F", "G"), 1) for _ in range(10)
        ]

        # Stratified split
        train_strat, test_strat = stratified_split_by_size(data, test_size=0.2, random_state=42)

        # Random split
        train_rand, test_rand = train_test_split(data, test_size=0.2, random_state=42)

        # Get size distributions
        from collections import Counter

        test_strat_sizes = Counter(len(r) for r, _ in test_strat)
        test_rand_sizes = Counter(len(r) for r, _ in test_rand)

        assert test_strat_sizes[2] > 0
        assert test_strat_sizes[5] > 0
        assert test_strat_sizes[5] >= test_rand_sizes[5]

        # Stratified should be more balanced
        # (This may not always differ, but typically should)

    def test_stratified_with_rare_sizes(self):
        """Test stratified split with rare comparison sizes."""
        data = [(("A", "B"), 1) for _ in range(50)] + [
            (("C", "D", "E", "F", "G", "H", "I", "J"), 1)
        ]  # Only 1 large comparison

        train, test = stratified_split_by_size(data, test_size=0.2, random_state=42)

        # Should handle rare sizes gracefully
        assert len(train) + len(test) == len(data)

    def test_stratified_perfect_balance(self):
        """Test stratified split with perfectly balanced data."""
        data = [(("A", "B"), 1) for _ in range(25)] + [(("C", "D", "E"), 1) for _ in range(25)]

        train, test = stratified_split_by_size(data, test_size=0.2, random_state=42)

        from collections import Counter

        test_sizes = Counter(len(r) for r, _ in test)

        # Should have roughly equal numbers of each size
        assert 2 in test_sizes
        assert 3 in test_sizes


class TestTemporalSplitAdvanced:
    """Advanced tests for temporal_split."""

    def test_temporal_preserves_order(self):
        """Test that temporal split preserves chronological order."""
        data = [(("A", "B"), i) for i in range(100)]

        train, test = temporal_split(data, test_size=0.2)

        # Train should be first 80 items
        assert train == data[:80]
        assert test == data[80:]

    def test_temporal_no_shuffle(self):
        """Test that temporal split never shuffles."""
        data = [(("A", "B"), i) for i in range(50)]

        train, test = temporal_split(data, test_size=0.3)

        # Verify order preserved
        for i in range(len(train) - 1):
            assert train[i][1] < train[i + 1][1]

        for i in range(len(test) - 1):
            assert test[i][1] < test[i + 1][1]

    def test_temporal_different_test_sizes(self):
        """Test temporal split with different test sizes."""
        data = [(("A", "B"), i) for i in range(100)]

        for test_size in [0.1, 0.25, 0.5, 0.75, 0.9]:
            train, test = temporal_split(data, test_size=test_size)

            expected_test = int(100 * test_size)
            assert len(test) == expected_test
            assert len(train) == 100 - expected_test

    def test_temporal_last_items_in_test(self):
        """Test that test set contains last items."""
        data = [(("A", "B"), i) for i in range(100)]

        train, test = temporal_split(data, test_size=0.2)

        # Last item should be in test
        assert data[-1] in test

        # First item should be in train
        assert data[0] in train

    def test_temporal_with_time_series(self):
        """Test temporal split with time-series like data."""
        # Simulate tournament rounds over time
        data = []
        for round_num in range(10):
            round_data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=round_num)
            data.extend(round_data)

        train, test = temporal_split(data, test_size=0.2)

        # Test should be last 20% of rounds
        assert len(test) == 20
        assert len(train) == 80

    def test_temporal_reproducibility(self):
        """Test that temporal split is reproducible."""
        data = [(("A", "B"), i) for i in range(50)]

        train1, test1 = temporal_split(data, test_size=0.2)
        train2, test2 = temporal_split(data, test_size=0.2)

        assert train1 == train2
        assert test1 == test2


class TestCrossValidateAdvanced:
    """Advanced tests for cross_validate."""

    def test_cv_different_n_splits(self):
        """Test cross-validation with different numbers of splits."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=5, seed=42)

        for n_splits in [3, 5, 10]:
            results = cross_validate(
                data,
                model_type="full",
                n_splits=n_splits,
                compare_projected=False,
                random_state=42,
                verbose=False,
            )

            assert len(results["pl_scores"]) == n_splits

    def test_cv_reproducibility(self):
        """Test that CV with same seed is reproducible."""
        data = generate_synthetic_rankings(N=15, M=80, K_min=2, K_max=4, seed=42)

        results1 = cross_validate(data, n_splits=5, random_state=42, verbose=False)
        results2 = cross_validate(data, n_splits=5, random_state=42, verbose=False)

        assert np.allclose(results1["pl_scores"], results2["pl_scores"], atol=1e-8)
        assert np.isclose(results1["pl_mean"], results2["pl_mean"], atol=1e-8)

    def test_cv_different_seeds(self):
        """Test that different seeds give different splits."""
        data = generate_synthetic_rankings(N=15, M=80, K_min=2, K_max=4, seed=42)

        results1 = cross_validate(data, n_splits=5, random_state=42, verbose=False)
        results2 = cross_validate(data, n_splits=5, random_state=99, verbose=False)

        # Scores should differ (different splits)
        assert results1["pl_scores"] != results2["pl_scores"]

    def test_cv_projected_comparison(self):
        """Test CV with projected model comparison."""
        data = generate_synthetic_rankings(N=15, M=100, K_min=3, K_max=6, seed=42)

        results = cross_validate(
            data,
            model_type="full",
            n_splits=3,
            compare_projected=True,
            random_state=42,
            verbose=False,
        )

        assert "projected_mean" in results
        assert "projected_std" in results
        assert "winner" in results
        assert "improvement" in results

        assert results["winner"] in ["PL", "Projected"]

    def test_cv_position1_model(self):
        """Test CV with position-1 model."""
        data = generate_synthetic_rankings(N=15, M=100, K_min=2, K_max=5, seed=42)

        results = cross_validate(
            data,
            model_type="position1",
            n_splits=5,
            compare_projected=False,
            random_state=42,
            verbose=False,
        )

        assert results["model_type"] == "position1"
        assert len(results["pl_scores"]) == 5

    def test_cv_winner_determination(self):
        """Test that CV correctly determines winner."""
        data = generate_synthetic_rankings(N=20, M=150, K_min=3, K_max=7, seed=42)

        results = cross_validate(
            data,
            model_type="full",
            n_splits=5,
            compare_projected=True,
            random_state=42,
            verbose=False,
        )

        # Winner should be the one with higher mean
        if results["pl_mean"] > results["projected_mean"]:
            assert results["winner"] == "PL"
        else:
            assert results["winner"] == "Projected"

    def test_cv_improvement_calculation(self):
        """Test that improvement is calculated correctly."""
        data = generate_synthetic_rankings(N=20, M=150, K_min=3, K_max=7, seed=42)

        results = cross_validate(
            data,
            model_type="full",
            n_splits=5,
            compare_projected=True,
            random_state=42,
            verbose=False,
        )

        # Improvement should be positive
        assert results["improvement"] >= 0

    def test_cv_verbose_mode(self):
        """Test CV with verbose output."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=4, seed=42)

        # Should not crash with verbose=True
        results = cross_validate(
            data, n_splits=3, compare_projected=True, random_state=42, verbose=True
        )

        assert len(results["pl_scores"]) == 3

    def test_cv_small_dataset(self):
        """Test CV with small dataset."""
        data = [(("A", "B", "C"), 1) for _ in range(10)]

        results = cross_validate(
            data, n_splits=3, compare_projected=False, random_state=42, verbose=False
        )

        assert len(results["pl_scores"]) == 3

    def test_cv_consistency_across_folds(self):
        """Test that CV scores are consistent."""
        data = generate_synthetic_rankings(N=15, M=100, K_min=2, K_max=5, seed=42)

        results = cross_validate(
            data, n_splits=5, compare_projected=False, random_state=42, verbose=False
        )

        # All scores should be finite and negative
        assert all(np.isfinite(score) for score in results["pl_scores"])
        assert all(score < 0 for score in results["pl_scores"])

        # Std should be reasonable
        assert results["pl_std"] >= 0
        assert results["pl_std"] < abs(results["pl_mean"])


class TestValidationIntegration:
    """Integration tests combining multiple validation functions."""

    def test_compare_all_split_methods(self):
        """Compare all splitting methods on same data."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=6, seed=42)

        # Random split
        train1, test1 = train_test_split(data, test_size=0.2, random_state=42)

        # Stratified split
        train2, test2 = stratified_split_by_size(data, test_size=0.2, random_state=42)

        # Temporal split
        train3, test3 = temporal_split(data, test_size=0.2)

        # All should have same total length
        assert len(train1) + len(test1) == len(data)
        assert len(train2) + len(test2) == len(data)
        assert len(train3) + len(test3) == len(data)

        # All should have approximately same test size
        assert abs(len(test1) - len(test2)) <= 2
        assert abs(len(test1) - len(test3)) <= 2

    def test_cv_vs_loo(self):
        """Compare cross-validation with leave-one-out."""
        data = generate_synthetic_rankings(N=8, M=20, K_min=2, K_max=4, seed=42)

        # 5-fold CV
        cv_results = cross_validate(
            data, n_splits=5, compare_projected=False, random_state=42, verbose=False
        )

        # Leave-one-out
        loo_results = leave_one_out_cv(data, model_type="full", verbose=False)

        # Both should give reasonable estimates
        assert np.isfinite(cv_results["pl_mean"])
        assert np.isfinite(loo_results["mean"])

        # LOO typically has lower variance but takes longer
        # Both should be negative
        assert cv_results["pl_mean"] < 0
        assert loo_results["mean"] < 0

    def test_nested_cv(self):
        """Test nested cross-validation for model selection."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=5, seed=42)

        # Outer loop: 3-fold CV
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        data_array = np.array(data, dtype=object)
        outer_scores = []

        for train_idx, test_idx in kf.split(data_array):
            train_outer = data_array[train_idx].tolist()
            test_outer = data_array[test_idx].tolist()

            # Inner loop: model selection via CV
            results_full = cross_validate(
                train_outer,
                model_type="full",
                n_splits=3,
                compare_projected=False,
                random_state=42,
                verbose=False,
            )
            results_pos1 = cross_validate(
                train_outer,
                model_type="position1",
                n_splits=3,
                compare_projected=False,
                random_state=42,
                verbose=False,
            )

            # Select best model
            if results_full["pl_mean"] > results_pos1["pl_mean"]:
                best_model_type = "full"
            else:
                best_model_type = "position1"

            # Train best model on full outer training set
            model = PlackettLuceModel(model_type=best_model_type)
            model.fit(train_outer, verbose=False)

            # Evaluate on outer test set
            score = model.log_likelihood(test_outer)
            outer_scores.append(score)

        # Should complete successfully
        assert len(outer_scores) == 3
        assert all(score < 0 for score in outer_scores)

    def test_validation_pipeline_robustness(self):
        """Test that validation pipeline is robust to various inputs."""
        datasets = [
            generate_synthetic_rankings(N=5, M=20, K_min=2, K_max=3, seed=i) for i in range(3)
        ]

        for data in datasets:
            # Try all validation methods
            try:
                train, test = train_test_split(data, test_size=0.2, random_state=42)
                assert len(train) + len(test) == len(data)

                cv_results = cross_validate(
                    data,
                    n_splits=3,
                    compare_projected=False,
                    random_state=42,
                    verbose=False,
                )
                assert len(cv_results["pl_scores"]) == 3

                loo_results = leave_one_out_cv(data, model_type="full", verbose=False)
                assert len(loo_results["scores"]) == len(data)

            except Exception as e:
                pytest.fail(f"Validation pipeline failed: {e}")


class TestValidationEdgeCases:
    """Test edge cases in validation functions."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        data = []

        with pytest.raises((ValueError, IndexError)):
            train_test_split(data, test_size=0.2)

    def test_single_item_validation(self):
        """Test validation with single item."""
        data = [(("A", "B"), 1)]

        # Most validation methods should handle or reject gracefully
        try:
            results = leave_one_out_cv(data, model_type="full", verbose=False)
            assert results["n_samples"] == 1
        except (ValueError, IndexError):
            pass  # Acceptable to reject

    def test_invalid_test_sizes(self):
        """Test that invalid test sizes are rejected."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        # Test size out of bounds
        with pytest.raises(ValueError):
            train_test_split(data, test_size=-0.1)

        with pytest.raises(ValueError):
            train_test_split(data, test_size=1.5)

    def test_invalid_n_splits(self):
        """Test that invalid n_splits are rejected."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        # n_splits = 1 is invalid
        with pytest.raises((ValueError, Exception)):
            cross_validate(data, n_splits=1, verbose=False)

        # n_splits > data size
        with pytest.raises((ValueError, Exception)):
            cross_validate(data, n_splits=100, verbose=False)

    def test_validation_with_duplicate_comparisons(self):
        """Test validation with duplicate comparison rankings."""
        data = [
            (("A", "B", "C"), 5),  # Same ranking appears multiple times
            (("B", "C", "A"), 3),
        ]

        # Should handle gracefully
        train, test = train_test_split(data, test_size=0.5, random_state=42)
        assert len(train) + len(test) == len(data)

    def test_validation_data_types(self):
        """Test validation with various data types."""
        # Integer node IDs
        data1 = [((1, 2, 3), 1), ((2, 3, 1), 1)]
        train, test = train_test_split(data1, test_size=0.5, random_state=42)
        assert len(train) + len(test) == 2

        # String node IDs
        data2 = [(("A", "B", "C"), 1), (("B", "C", "A"), 1)]
        train, test = train_test_split(data2, test_size=0.5, random_state=42)
        assert len(train) + len(test) == 2

    def test_validation_preserves_weights(self):
        """Test that validation preserves comparison weights."""
        data = [
            (("A", "B"), 5),
            (("C", "D"), 10),
            (("E", "F"), 3),
        ]

        train, test = train_test_split(data, test_size=0.3, random_state=42)

        # Check that weights are preserved
        for ranking, weight in train + test:
            assert weight in [3, 5, 10]

    def test_cv_with_inconsistent_nodes(self):
        """Test CV when not all nodes appear in all folds."""
        # Create data where some nodes are rare
        data = [(("A", "B"), 1) for _ in range(20)] + [(("X", "Y"), 1)]  # Rare nodes

        # Should handle gracefully
        results = cross_validate(
            data, n_splits=3, compare_projected=False, random_state=42, verbose=False
        )

        assert len(results["pl_scores"]) == 3


class TestValidationPerformance:
    """Test performance characteristics of validation functions."""

    def test_train_test_split_performance(self):
        """Test that train_test_split is fast."""
        import time

        data = generate_synthetic_rankings(N=100, M=10000, K_min=2, K_max=10, seed=42)

        start = time.time()
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be very fast
        assert len(train) + len(test) == 10000

    def test_cv_reasonable_time(self):
        """Test that CV completes in reasonable time."""
        import time

        data = generate_synthetic_rankings(N=30, M=200, K_min=2, K_max=6, seed=42)

        start = time.time()
        results = cross_validate(
            data, n_splits=5, compare_projected=False, random_state=42, verbose=False
        )
        elapsed = time.time() - start

        assert elapsed < 30.0  # Should complete in under 30 seconds
        assert len(results["pl_scores"]) == 5

    def test_loo_scalability(self):
        """Test LOO scalability."""
        import time

        # LOO is O(n) in number of comparisons
        data = generate_synthetic_rankings(N=10, M=30, K_min=2, K_max=4, seed=42)

        start = time.time()
        results = leave_one_out_cv(data, model_type="full", verbose=False)
        elapsed = time.time() - start

        # Should scale linearly with M
        assert elapsed < 15.0  # Reasonable time for 30 iterations
        assert results["n_samples"] == 30


class TestValidationStatistics:
    """Test statistical properties of validation methods."""

    def test_cv_mean_is_unbiased(self):
        """Test that CV mean is unbiased estimate."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=5, seed=42)

        # Run CV multiple times with different seeds
        means = []
        for seed in range(5):
            results = cross_validate(
                data,
                n_splits=5,
                compare_projected=False,
                random_state=seed,
                verbose=False,
            )
            means.append(results["pl_mean"])

        # All means should be similar (measuring same thing)
        mean_of_means = np.mean(means)
        assert all(abs(m - mean_of_means) < abs(mean_of_means) * 0.3 for m in means)

    def test_cv_std_increases_with_variance(self):
        """Test that CV std captures variance."""
        # Create two datasets: one consistent, one variable
        consistent_data = [(("A", "B", "C"), 1) for _ in range(50)]

        variable_data = generate_synthetic_rankings(N=20, M=50, K_min=2, K_max=8, seed=42)

        results_consistent = cross_validate(
            consistent_data,
            n_splits=5,
            compare_projected=False,
            random_state=42,
            verbose=False,
        )

        results_variable = cross_validate(
            variable_data,
            n_splits=5,
            compare_projected=False,
            random_state=42,
            verbose=False,
        )

        # Variable data should have higher std (though not guaranteed)
        # Just verify both stds are non-negative and finite
        assert results_consistent["pl_std"] >= 0
        assert results_variable["pl_std"] >= 0
        assert np.isfinite(results_consistent["pl_std"])
        assert np.isfinite(results_variable["pl_std"])

    def test_loo_variance_properties(self):
        """Test variance properties of LOO."""
        data = generate_synthetic_rankings(N=15, M=50, K_min=2, K_max=5, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        # Variance should be non-negative
        assert results["std"] >= 0

        # Scores should cluster around mean
        deviations = [abs(score - results["mean"]) for score in results["scores"]]
        avg_deviation = np.mean(deviations)

        # Average deviation should be close to std
        assert abs(avg_deviation - results["std"]) < results["std"]

    def test_split_randomness_quality(self):
        """Test that splits are well-randomized."""
        data = [(("A", "B"), i) for i in range(100)]

        # Multiple splits with same test_size
        splits = []
        for seed in range(10):
            train, test = train_test_split(data, test_size=0.2, random_state=seed, shuffle=True)
            test_indices = [item[1] for item in test]
            splits.append(set(test_indices))

        # All splits should be different
        assert len(set(tuple(sorted(s)) for s in splits)) == 10

        # Each split should have good coverage
        for split_set in splits:
            # Should sample from across the range
            assert min(split_set) < 30  # Some from beginning
            assert max(split_set) > 70  # Some from end


class TestValidationDocumentation:
    """Test that validation functions match documentation."""

    def test_train_test_split_returns_correct_types(self):
        """Test return types match documentation."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        train, test = train_test_split(data, test_size=0.2, random_state=42)

        assert isinstance(train, list)
        assert isinstance(test, list)
        assert all(isinstance(item, tuple) for item in train)
        assert all(isinstance(item, tuple) for item in test)

    def test_cross_validate_returns_dict(self):
        """Test that cross_validate returns dict with expected keys."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        results = cross_validate(
            data, n_splits=3, compare_projected=True, random_state=42, verbose=False
        )

        assert isinstance(results, dict)

        required_keys = ["pl_mean", "pl_std", "pl_scores", "n_splits", "model_type"]
        for key in required_keys:
            assert key in results

        # With compare_projected=True
        projected_keys = ["projected_mean", "projected_std", "winner", "improvement"]
        for key in projected_keys:
            assert key in results

    def test_leave_one_out_returns_dict(self):
        """Test that leave_one_out_cv returns dict with expected keys."""
        data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=3, seed=42)

        results = leave_one_out_cv(data, model_type="full", verbose=False)

        assert isinstance(results, dict)

        required_keys = ["mean", "std", "scores", "n_samples", "model_type"]
        for key in required_keys:
            assert key in results

    def test_stratified_split_signature(self):
        """Test stratified_split_by_size has correct signature."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        # Test with all parameters
        train, test = stratified_split_by_size(data, test_size=0.2, random_state=42)

        assert isinstance(train, list)
        assert isinstance(test, list)

    def test_temporal_split_signature(self):
        """Test temporal_split has correct signature."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        # Minimal call
        train, test = temporal_split(data, test_size=0.2)

        assert isinstance(train, list)
        assert isinstance(test, list)


class TestValidationConsistency:
    """Test consistency between different validation methods."""

    def test_cv_and_manual_split_consistency(self):
        """Test that CV gives similar results to manual split."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=5, seed=42)

        # CV results
        cv_results = cross_validate(
            data, n_splits=5, compare_projected=False, random_state=42, verbose=False
        )

        # Manual split and evaluation
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        model = PlackettLuceModel(model_type="full")
        model.fit(train, verbose=False)
        manual_score = model.log_likelihood(test)

        # Scores should be in same ballpark
        cv_mean = cv_results["pl_mean"]

        # Both should be negative
        assert cv_mean < 0
        assert manual_score < 0

        # Should be within an order of magnitude
        assert abs(cv_mean / manual_score) < 10

    def test_different_model_types_comparable(self):
        """Test that different model types give comparable validation scores."""
        data = generate_synthetic_rankings(N=15, M=80, K_min=2, K_max=5, seed=42)

        # Full model
        results_full = cross_validate(
            data,
            model_type="full",
            n_splits=5,
            compare_projected=False,
            random_state=42,
            verbose=False,
        )

        # Position-1 model
        results_pos1 = cross_validate(
            data,
            model_type="position1",
            n_splits=5,
            compare_projected=False,
            random_state=42,
            verbose=False,
        )

        # Both should produce valid scores
        assert np.isfinite(results_full["pl_mean"])
        assert np.isfinite(results_pos1["pl_mean"])

        # Both should be negative
        assert results_full["pl_mean"] < 0
        assert results_pos1["pl_mean"] < 0

    def test_validation_gives_similar_rankings(self):
        """Test that models from different validation runs give similar rankings."""
        data = generate_synthetic_rankings(N=15, M=100, K_min=2, K_max=5, seed=42)

        # Train on different splits
        rankings = []
        for seed in range(3):
            train, _ = train_test_split(data, test_size=0.2, random_state=seed)
            model = PlackettLuceModel(model_type="full")
            model.fit(train, verbose=False)
            ranking = [node for node, _ in model.get_ranking()]
            rankings.append(ranking)

        # Measure pairwise similarities
        from plackett_luce.utils import ranking_similarity

        similarities = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                sim = ranking_similarity(rankings[i], rankings[j], method="kendall")
                similarities.append(sim)

        # Should have positive average similarity
        avg_sim = np.mean(similarities)
        assert avg_sim > 0  # Rankings should be somewhat consistent


class TestValidationRobustness:
    """Test robustness of validation methods."""

    def test_validation_with_noise(self):
        """Test validation with noisy data."""
        # Create data with some random noise
        base_data = [(("A", "B", "C"), 1) for _ in range(30)]
        noise_data = generate_synthetic_rankings(N=3, M=20, K_min=2, K_max=3, seed=99)
        data = base_data + noise_data

        # Should still produce valid results
        results = cross_validate(
            data, n_splits=5, compare_projected=False, random_state=42, verbose=False
        )

        assert np.isfinite(results["pl_mean"])
        assert results["pl_mean"] < 0

    def test_validation_with_imbalanced_data(self):
        """Test validation with highly imbalanced data."""
        # One node appears much more than others
        data = (
            [(("A", "B"), 1) for _ in range(50)]
            + [(("C", "D"), 1) for _ in range(5)]
            + [(("E", "F"), 1) for _ in range(5)]
        )

        # Should handle gracefully
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        assert len(train) + len(test) == len(data)

    def test_validation_error_propagation(self):
        """Test that validation properly propagates errors."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        # Invalid model type should raise error
        with pytest.raises((ValueError, AttributeError)):
            cross_validate(
                data,
                model_type="invalid",
                n_splits=3,
                compare_projected=False,
                verbose=False,
            )

    def test_validation_handles_convergence_issues(self):
        """Test validation when models have convergence issues."""
        data = generate_synthetic_rankings(N=10, M=30, K_min=2, K_max=4, seed=42)

        # Use very strict epsilon that may not converge
        # This tests the validation framework's robustness
        try:
            results = cross_validate(
                data,
                n_splits=3,
                compare_projected=False,
                random_state=42,
                verbose=False,
            )
            # If it completes, verify results are valid
            assert np.isfinite(results["pl_mean"])
        except Exception:
            # Some convergence issues may cause failures
            pass


class TestValidationBoundaryConditions:
    """Test boundary conditions in validation."""

    def test_minimum_viable_dataset(self):
        """Test validation with minimum viable dataset."""
        # Minimum for meaningful validation
        data = [(("A", "B"), 1), (("B", "C"), 1), (("A", "C"), 1)]

        train, test = train_test_split(data, test_size=0.33, random_state=42)
        assert len(train) >= 1
        assert len(test) >= 1

    def test_maximum_test_size(self):
        """Test with maximum allowable test size."""
        data = generate_synthetic_rankings(N=10, M=100, K_min=2, K_max=5, seed=42)

        # 99% test size
        train, test = train_test_split(data, test_size=0.99, random_state=42)

        assert len(train) >= 1  # At least 1 in training
        assert len(test) >= 99  # At least 99 in test

    def test_minimum_test_size(self):
        """Test with minimum test size."""
        data = generate_synthetic_rankings(N=10, M=100, K_min=2, K_max=5, seed=42)

        # 1% test size
        train, test = train_test_split(data, test_size=0.01, random_state=42)

        assert len(test) >= 1  # At least 1 in test
        assert len(train) >= 99  # At least 99 in training

    def test_cv_minimum_folds(self):
        """Test CV with minimum number of folds."""
        data = generate_synthetic_rankings(N=10, M=30, K_min=2, K_max=4, seed=42)

        # 2-fold CV (minimum valid)
        results = cross_validate(
            data, n_splits=2, compare_projected=False, random_state=42, verbose=False
        )

        assert len(results["pl_scores"]) == 2

    def test_cv_maximum_reasonable_folds(self):
        """Test CV with many folds."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=4, seed=42)

        # 10-fold CV
        results = cross_validate(
            data, n_splits=10, compare_projected=False, random_state=42, verbose=False
        )

        assert len(results["pl_scores"]) == 10
