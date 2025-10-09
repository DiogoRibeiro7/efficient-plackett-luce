"""
Tests for core Plackett-Luce model functionality.
"""

import numpy as np
import pytest

import plackett_luce.core as core
from plackett_luce import PlackettLuceModel
from plackett_luce.utils import generate_synthetic_rankings


class TestPlackettLuceModel:
    """Test suite for PlackettLuceModel class."""

    def test_initialization(self):
        """Test model initialization with different parameters."""
        model = PlackettLuceModel(model_type="full")
        assert model.model_type == "full"
        assert model.epsilon == 1e-6
        assert not model.is_fitted

        model_pos1 = PlackettLuceModel(model_type="position1", epsilon=1e-4)
        assert model_pos1.model_type == "position1"
        assert model_pos1.epsilon == 1e-4

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            PlackettLuceModel(model_type="invalid")

    def test_preprocess_deterministic_mixed_types(self):
        """Mixed-type nodes should sort deterministically and without duplicates."""

        model = PlackettLuceModel(model_type="full")
        hyperedges = [((1, "2", 3.0), 1.0)]

        N, edges, weights, edge_lengths, edge_starts = model._preprocess_edges(hyperedges)

        assert N == 3
        assert list(model.node_to_idx.keys()) == [1, "2", 3.0]
        assert list(model.idx_to_node.values()) == [1, "2", 3.0]
        assert edges.tolist() == [0, 1, 2]
        assert weights.tolist() == [1.0]
        assert edge_lengths.tolist() == [3]
        assert edge_starts.tolist() == [0]

    def test_fit_simple_data(self):
        """Test fitting on simple data."""
        data = [
            (("A", "B", "C"), 1),
            (("B", "C", "A"), 1),
            (("A", "B"), 2),
        ]

        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        assert model.is_fitted
        assert "iterations" in stats
        assert "time" in stats
        assert stats["iterations"] > 0

    def test_fit_convergence(self):
        """Test that model converges."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=5, seed=42)

        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        assert stats["iterations"] < model.max_iterations
        assert model.scores is not None
        assert len(model.scores) == 20
        assert model.converged is True

    def test_fit_reproducibility_with_seed(self):
        """Fitting with the same seed should produce identical scores."""
        data = generate_synthetic_rankings(N=12, M=60, K_min=2, K_max=5, seed=101)

        model1 = PlackettLuceModel(model_type="full", random_state=123, max_iterations=500)
        stats1 = model1.fit(data, verbose=False)

        model2 = PlackettLuceModel(model_type="full", random_state=123, max_iterations=500)
        stats2 = model2.fit(data, verbose=False)

        assert np.allclose(model1.scores, model2.scores, atol=1e-8)
        assert stats1["iterations"] == stats2["iterations"]
        assert stats1["converged"] == stats2["converged"]
        assert model1.converged == model2.converged == stats1["converged"]

    def test_get_ranking(self):
        """Test getting rankings."""
        data = [(("A", "B", "C"), 2), (("A", "C"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()

        assert len(ranking) == 3
        assert ranking[0][0] == "A"  # A should be top
        assert all(isinstance(score, float) for _, score in ranking)

    def test_get_ranking_top_k(self):
        """Test getting top-k rankings."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        top_3 = model.get_ranking(top_k=3)

        assert len(top_3) == 3
        assert top_3[0][1] >= top_3[1][1] >= top_3[2][1]

    def test_predict_probability(self):
        """Test probability prediction."""
        data = [(("A", "B", "C"), 1), (("A", "B"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        prob = model.predict_probability(("A", "B", "C"))

        assert 0 <= prob <= 1
        assert isinstance(prob, float)

    def test_predict_probability_unknown_node(self):
        """Test that predicting with unknown node raises error."""
        data = [(("A", "B"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        with pytest.raises(ValueError):
            model.predict_probability(("A", "Z"))

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        ll = model.log_likelihood(data)

        assert isinstance(ll, float)
        assert ll < 0  # Log-likelihood should be negative

    def test_save_load(self, tmp_path):
        """Test saving and loading models."""
        data = [(("A", "B", "C"), 1), (("B", "C"), 1)]

        model = PlackettLuceModel(model_type="full", use_cache=False, timeout=5.0)
        model.fit(data, verbose=False)

        # Save
        filepath = tmp_path / "test_model.pkl"
        model.save(filepath)

        # Load
        loaded_model = PlackettLuceModel.load(filepath)

        # Compare
        assert loaded_model.model_type == model.model_type
        assert np.allclose(loaded_model.scores, model.scores)
        assert loaded_model.node_to_idx == model.node_to_idx
        assert loaded_model.use_cache is False
        assert loaded_model.timeout == 5.0

    def test_position1_model(self):
        """Test position-1-breaking model."""
        data = [(("A", "B", "C"), 2), (("B", "A", "C"), 1)]

        model = PlackettLuceModel(model_type="position1")
        stats = model.fit(data, verbose=False)

        assert model.is_fitted
        assert stats["iterations"] > 0

        prob = model.predict_probability(("A", "B", "C"))
        assert 0 <= prob <= 1

    def test_empty_data(self):
        """Test that empty data raises error."""
        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit([], verbose=False)

    def test_invalid_data_format(self):
        """Test that invalid data format raises error."""
        model = PlackettLuceModel(model_type="full")

        with pytest.raises(ValueError):
            model.fit([("A", "B")], verbose=False)  # Missing weight

    def test_score_normalization(self):
        """Test that scores are properly normalized."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # Geometric mean should be close to 1
        geometric_mean = np.exp(np.mean(np.log(model.scores)))
        assert np.isclose(geometric_mean, 1.0, atol=1e-6)

    def test_warning_small_dataset(self):
        """Dataset with fewer than 10 comparisons should trigger warning."""
        data = [
            (("A", "B"), 1),
            (("B", "A"), 1),
            (("A", "B"), 1),
            (("C", "D"), 1),
            (("D", "C"), 1),
            (("C", "D"), 1),
            (("E", "A"), 1),
            (("E", "C"), 1),
            (("E", "B"), 1),
        ]  # 9 comparisons

        model = PlackettLuceModel(model_type="full")
        with pytest.warns(UserWarning, match="Dataset is very small"):
            model.fit(data, verbose=False)

    def test_warning_rare_nodes(self):
        """Nodes appearing in fewer than 3 comparisons should trigger warning."""
        common_comparisons = [
            (("A", "B", "C"), 1),
            (("B", "C", "D"), 1),
            (("C", "D", "E"), 1),
            (("A", "D", "E"), 1),
            (("A", "B", "E"), 1),
            (("B", "C", "E"), 1),
            (("A", "C", "D"), 1),
            (("B", "D", "E"), 1),
        ]
        rare_comparisons = [
            (("Z", "A", "B"), 1),
            (("Z", "C", "D"), 1),
        ]
        data = common_comparisons + rare_comparisons  # 10 comparisons

        model = PlackettLuceModel(model_type="full")
        with pytest.warns(UserWarning, match="fewer than 3 comparisons"):
            model.fit(data, verbose=False)

    def test_warning_large_comparison(self):
        """Very large comparison sizes should trigger warning."""
        data = [(("A", "B", "C", "D"), 1) for _ in range(12)]

        model = PlackettLuceModel(model_type="full")
        with pytest.warns(UserWarning, match="Large comparison sizes"):
            model.fit(data, verbose=False)

    def test_fit_max_iterations_warning(self):
        """Hitting max iterations should warn and mark model as not converged."""
        data = generate_synthetic_rankings(N=6, M=30, K_min=2, K_max=4, seed=2024)

        model = PlackettLuceModel(
            model_type="full", epsilon=1e-12, max_iterations=1, random_state=777
        )
        with pytest.warns(UserWarning, match="Maximum iterations"):
            stats = model.fit(data, verbose=False)

        assert stats["converged"] is False
        assert stats["timed_out"] is False
        assert stats["iterations"] == model.max_iterations
        assert model.converged is False

    def test_fit_timeout(self, monkeypatch):
        """Timeout should abort fitting and emit warning."""
        data = generate_synthetic_rankings(N=8, M=40, seed=3)

        times = [0.0, 2.0]

        def fake_time():
            return times.pop(0) if times else 2.0

        monkeypatch.setattr(core.time, "time", fake_time)

        model = PlackettLuceModel(model_type="full", timeout=1.0)
        with pytest.warns(UserWarning, match="Timeout reached"):
            stats = model.fit(data, verbose=False)

        assert stats["timed_out"] is True
        assert stats["converged"] is False
        assert stats["iterations"] <= model.max_iterations
