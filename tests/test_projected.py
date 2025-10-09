"""
Tests for projected Plackett-Luce models.
"""

import numpy as np

from plackett_luce import PlackettLuceModel, ProjectedPlackettLuce
from plackett_luce.utils import generate_synthetic_rankings


class TestProjectedPlackettLuce:
    """Test suite for ProjectedPlackettLuce class."""

    def test_initialization(self):
        """Test model initialization."""
        model = ProjectedPlackettLuce(model_type="full")
        assert model.model_type == "full"
        assert isinstance(model, PlackettLuceModel)
        assert not model.is_fitted

    def test_initialization_position1(self):
        """Test position-1 variant initialization."""
        model = ProjectedPlackettLuce(model_type="position1")
        assert model.model_type == "position1"

    def test_project_to_pairwise_full(self):
        """Test projection to pairwise for full model."""
        model = ProjectedPlackettLuce(model_type="full")

        # Single 3-way comparison
        hyperedges = [(("A", "B", "C"), 1)]
        pairwise = model._project_to_pairwise(hyperedges)

        # Should create 3 pairwise comparisons: A>B, A>C, B>C
        assert len(pairwise) == 3

        pairwise_set = {ranking for ranking, _ in pairwise}
        expected = {("A", "B"), ("A", "C"), ("B", "C")}
        assert pairwise_set == expected

    def test_project_to_pairwise_position1(self):
        """Test projection to pairwise for position-1 model."""
        model = ProjectedPlackettLuce(model_type="position1")

        # Single 3-way comparison
        hyperedges = [(("A", "B", "C"), 1)]
        pairwise = model._project_to_pairwise(hyperedges)

        # Should create 2 pairwise comparisons: A>B, A>C (first vs rest)
        assert len(pairwise) == 2

        pairwise_set = {ranking for ranking, _ in pairwise}
        expected = {("A", "B"), ("A", "C")}
        assert pairwise_set == expected

    def test_project_preserves_weights(self):
        """Test that projection preserves weights."""
        model = ProjectedPlackettLuce(model_type="full")

        hyperedges = [(("A", "B", "C"), 3)]
        pairwise = model._project_to_pairwise(hyperedges)

        # All projected pairs should have weight 3
        for _, weight in pairwise:
            assert weight == 3

    def test_project_multiple_edges(self):
        """Test projection with multiple edges."""
        model = ProjectedPlackettLuce(model_type="full")

        hyperedges = [(("A", "B", "C"), 1), (("D", "E"), 1)]
        pairwise = model._project_to_pairwise(hyperedges)

        # First edge: 3 pairs, second edge: 1 pair
        assert len(pairwise) == 4

    def test_project_pairwise_unchanged(self):
        """Test that pairwise comparisons remain unchanged."""
        model = ProjectedPlackettLuce(model_type="full")

        hyperedges = [(("A", "B"), 1)]
        pairwise = model._project_to_pairwise(hyperedges)

        # Should be identical
        assert len(pairwise) == 1
        assert pairwise[0] == (("A", "B"), 1)

    def test_fit_on_multibody_data(self):
        """Test fitting on multi-body data."""
        data = [(("A", "B", "C"), 1), (("B", "C", "A"), 1), (("C", "A", "B"), 1)]

        model = ProjectedPlackettLuce(model_type="full")
        stats = model.fit(data, verbose=False)

        assert model.is_fitted
        assert stats["iterations"] > 0
        assert len(model.scores) == 3

    def test_fit_position1_projection(self):
        """Test fitting with position-1 projection."""
        data = [(("A", "B", "C"), 2), (("B", "A", "C"), 1)]

        model = ProjectedPlackettLuce(model_type="position1")
        model.fit(data, verbose=False)

        assert model.is_fitted
        # A wins twice, B wins once, so A should score higher
        score_a = model.get_score("A")
        score_b = model.get_score("B")
        assert score_a > score_b

    def test_ranking_after_projection(self):
        """Test that ranking works after projection."""
        data = [(("A", "B", "C", "D"), 1), (("A", "B"), 1)]

        model = ProjectedPlackettLuce(model_type="full")
        model.fit(data, verbose=False)

        ranking = model.get_ranking()
        assert len(ranking) == 4
        assert ranking[0][0] == "A"  # A consistently ranks first

    def test_predict_probability_after_projection(self):
        """Test probability prediction after projection."""
        data = [(("A", "B", "C"), 1), (("A", "B"), 1)]

        model = ProjectedPlackettLuce(model_type="full")
        model.fit(data, verbose=False)

        prob = model.predict_probability(("A", "B"))
        assert 0 <= prob <= 1

    def test_log_likelihood_after_projection(self):
        """Test log-likelihood computation after projection."""
        data = generate_synthetic_rankings(N=10, M=50, K_min=2, K_max=5, seed=42)

        model = ProjectedPlackettLuce(model_type="full")
        model.fit(data, verbose=False)

        ll = model.log_likelihood(data)
        assert isinstance(ll, float)
        assert ll < 0

    def test_comparison_with_full_model(self):
        """Test that projected model differs from full model."""
        data = [(("A", "B", "C"), 1), (("B", "C", "A"), 1), (("C", "A", "B"), 1)]

        # Full multi-body model
        model_full = PlackettLuceModel(model_type="full")
        model_full.fit(data, verbose=False)

        # Projected model
        model_proj = ProjectedPlackettLuce(model_type="full")
        model_proj.fit(data, verbose=False)

        # Scores should be different
        scores_full = model_full.scores
        scores_proj = model_proj.scores

        # Not exactly equal (unless data is very simple)
        assert not np.allclose(scores_full, scores_proj, atol=1e-6)

    def test_save_load_projected_model(self, tmp_path):
        """Test saving and loading projected models."""
        data = [(("A", "B", "C"), 1), (("B", "C"), 1)]

        model = ProjectedPlackettLuce(model_type="full")
        model.fit(data, verbose=False)

        # Save
        filepath = tmp_path / "projected_model.pkl"
        model.save(filepath)

        # Load (use base class load method)
        loaded_model = PlackettLuceModel.load(filepath)

        # Verify
        assert loaded_model.model_type == model.model_type
        assert np.allclose(loaded_model.scores, model.scores)

    def test_projection_with_large_K(self):
        """Test projection with large comparison sizes."""
        model = ProjectedPlackettLuce(model_type="full")

        # 10-way comparison
        nodes = [f"Node{i}" for i in range(10)]
        hyperedges = [(tuple(nodes), 1)]

        pairwise = model._project_to_pairwise(hyperedges)

        # Should create C(10,2) = 45 pairs
        assert len(pairwise) == 45

    def test_projection_consistency(self):
        """Test that projection is deterministic."""
        data = [(("A", "B", "C"), 1), (("D", "E", "F"), 1)]

        model1 = ProjectedPlackettLuce(model_type="full")
        model2 = ProjectedPlackettLuce(model_type="full")

        pairwise1 = model1._project_to_pairwise(data)
        pairwise2 = model2._project_to_pairwise(data)

        assert pairwise1 == pairwise2

    def test_empty_projection(self):
        """Test projection with empty data."""
        model = ProjectedPlackettLuce(model_type="full")

        pairwise = model._project_to_pairwise([])
        assert pairwise == []

    def test_synthetic_data_projection(self):
        """Test projection on synthetic data."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=3, K_max=8, seed=42)

        model = ProjectedPlackettLuce(model_type="full")
        stats = model.fit(data, verbose=False)

        assert model.is_fitted
        assert stats["iterations"] < model.max_iterations
        assert len(model.scores) == 20


class TestProjectionProperties:
    """Test mathematical properties of projections."""

    def test_projection_transitivity(self):
        """Test that full projection respects transitivity."""
        model = ProjectedPlackettLuce(model_type="full")

        # A > B > C should give A>B, A>C, B>C
        hyperedges = [(("A", "B", "C"), 1)]
        pairwise = model._project_to_pairwise(hyperedges)

        # Check all expected pairs exist
        pairs_dict = {ranking: weight for ranking, weight in pairwise}

        assert ("A", "B") in pairs_dict
        assert ("A", "C") in pairs_dict
        assert ("B", "C") in pairs_dict

        # No reverse pairs
        assert ("B", "A") not in pairs_dict
        assert ("C", "A") not in pairs_dict
        assert ("C", "B") not in pairs_dict

    def test_projection_count_formula_full(self):
        """Test that full projection creates correct number of pairs."""
        model = ProjectedPlackettLuce(model_type="full")

        for K in range(2, 10):
            nodes = [f"N{i}" for i in range(K)]
            hyperedges = [(tuple(nodes), 1)]

            pairwise = model._project_to_pairwise(hyperedges)

            # Should create K*(K-1)/2 pairs
            expected_pairs = K * (K - 1) // 2
            assert len(pairwise) == expected_pairs

    def test_projection_count_formula_position1(self):
        """Test that position-1 projection creates correct number of pairs."""
        model = ProjectedPlackettLuce(model_type="position1")

        for K in range(2, 10):
            nodes = [f"N{i}" for i in range(K)]
            hyperedges = [(tuple(nodes), 1)]

            pairwise = model._project_to_pairwise(hyperedges)

            # Should create K-1 pairs (first vs all others)
            expected_pairs = K - 1
            assert len(pairwise) == expected_pairs

    def test_weight_aggregation(self):
        """Test that repeated projections aggregate correctly."""
        model = ProjectedPlackettLuce(model_type="full")

        # Two identical rankings
        hyperedges = [(("A", "B"), 2), (("C", "D"), 3)]

        pairwise = model._project_to_pairwise(hyperedges)

        pairs_dict = {ranking: weight for ranking, weight in pairwise}

        assert pairs_dict[("A", "B")] == 2
        assert pairs_dict[("C", "D")] == 3
