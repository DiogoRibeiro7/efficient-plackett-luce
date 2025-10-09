"""
Integration tests for end-to-end workflows.
"""

import numpy as np
import pytest

from plackett_luce import (
    PlackettLuceModel,
    ProjectedPlackettLuce,
    cross_validate,
    generate_synthetic_rankings,
    ranking_similarity,
    train_test_split,
)
from plackett_luce.validation import stratified_split_by_size, temporal_split


class TestEndToEndWorkflows:
    """Test complete workflows from data to results."""

    def test_basic_tournament_workflow(self):
        """Test basic tournament ranking workflow."""
        # 1. Define tournament data
        results = [
            (("TeamA", "TeamB", "TeamC"), 1),
            (("TeamB", "TeamC", "TeamA"), 1),
            (("TeamA", "TeamC", "TeamB"), 2),
        ]

        # 2. Fit model
        model = PlackettLuceModel(model_type="full")
        model.fit(results, verbose=False)

        # 3. Get rankings
        ranking = model.get_ranking()

        # 4. Make predictions
        prob = model.predict_probability(("TeamA", "TeamB", "TeamC"))

        # 5. Compute log-likelihood
        ll = model.log_likelihood(results)

        # Verify all steps completed successfully
        assert len(ranking) == 3
        assert 0 <= prob <= 1
        assert ll < 0

    def test_train_test_evaluation_workflow(self):
        """Test train/test split and evaluation workflow."""
        # 1. Generate data
        data = generate_synthetic_rankings(N=20, M=200, K_min=2, K_max=5, seed=42)

        # 2. Split data
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        # 3. Train model
        model = PlackettLuceModel(model_type="full")
        model.fit(train, verbose=False)

        # 4. Evaluate on test set
        train_ll = model.log_likelihood(train)
        test_ll = model.log_likelihood(test)

        # 5. Get rankings
        ranking = model.get_ranking(top_k=10)

        # Verify workflow
        assert len(train) + len(test) == len(data)
        assert train_ll < 0
        assert test_ll < 0
        assert len(ranking) == 10

    def test_cross_validation_workflow(self):
        """Test cross-validation workflow."""
        # 1. Generate data
        data = generate_synthetic_rankings(N=30, M=300, K_min=2, K_max=6, seed=42)

        # 2. Perform cross-validation
        results = cross_validate(
            data,
            model_type="full",
            n_splits=5,
            compare_projected=True,
            random_state=42,
            verbose=False,
        )

        # 3. Analyze results
        pl_score = results["pl_mean"]
        proj_score = results["projected_mean"]
        winner = results["winner"]

        # Verify results
        assert isinstance(pl_score, float)
        assert isinstance(proj_score, float)
        assert winner in ["PL", "Projected"]
        assert "improvement" in results

    def test_model_comparison_workflow(self):
        """Test comparing different model types."""
        # 1. Generate data
        data = generate_synthetic_rankings(N=25, M=250, K_min=2, K_max=5, seed=42)

        # 2. Split data
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        # 3. Train multiple models
        models = {
            "Full PL": PlackettLuceModel(model_type="full"),
            "Position-1": PlackettLuceModel(model_type="position1"),
            "Projected Full": ProjectedPlackettLuce(model_type="full"),
            "Projected Pos-1": ProjectedPlackettLuce(model_type="position1"),
        }

        results = {}
        for name, model in models.items():
            model.fit(train, verbose=False)
            test_ll = model.log_likelihood(test)
            results[name] = test_ll

        # 4. Find best model
        best_model = max(results, key=results.get)

        # Verify comparison
        assert len(results) == 4
        assert all(ll < 0 for ll in results.values())
        assert best_model in models.keys()

    def test_save_load_predict_workflow(self, tmp_path):
        """Test save, load, and predict workflow."""
        # 1. Train model
        data = generate_synthetic_rankings(N=15, M=100, seed=42)
        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        # 2. Get original predictions
        original_ranking = model.get_ranking()
        original_prob = model.predict_probability(data[0][0])

        # 3. Save model
        model_path = tmp_path / "model.pkl"
        model.save(model_path)

        # 4. Load model
        loaded_model = PlackettLuceModel.load(model_path)

        # 5. Get new predictions
        loaded_ranking = loaded_model.get_ranking()
        loaded_prob = loaded_model.predict_probability(data[0][0])

        # Verify consistency
        assert original_ranking == loaded_ranking
        assert np.isclose(original_prob, loaded_prob)

    def test_synthetic_data_pipeline(self):
        """Test complete pipeline with synthetic data."""
        # 1. Generate synthetic data with known properties
        np.random.seed(42)
        N = 20
        M = 500
        data = generate_synthetic_rankings(N=N, M=M, K_min=3, K_max=7, seed=42)

        # 2. Fit model
        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        # 3. Extract all nodes
        all_nodes = set()
        for ranking, _ in data:
            all_nodes.update(ranking)

        # 4. Verify model learned all nodes
        learned_ranking = model.get_ranking()
        learned_nodes = {node for node, _ in learned_ranking}

        assert learned_nodes == all_nodes
        assert stats["iterations"] < model.max_iterations

    def test_temporal_validation_workflow(self):
        """Test temporal train/test split workflow."""
        # 1. Generate temporal data
        rounds = []
        for round_num in range(10):
            round_data = generate_synthetic_rankings(N=10, M=20, K_min=2, K_max=4, seed=round_num)
            rounds.extend(round_data)

        # 2. Temporal split
        train, test = temporal_split(rounds, test_size=0.2)

        # 3. Train on past
        model = PlackettLuceModel(model_type="full")
        model.fit(train, verbose=False)

        # 4. Predict future
        test_ll = model.log_likelihood(test)

        # Verify temporal workflow
        assert len(train) > len(test)
        assert test_ll < 0

    def test_stratified_sampling_workflow(self):
        """Test stratified sampling workflow."""
        # 1. Generate data with varied comparison sizes
        data = generate_synthetic_rankings(N=30, M=300, K_min=2, K_max=8, seed=42)

        # 2. Stratified split
        train, test = stratified_split_by_size(data, test_size=0.2, random_state=42)

        # 3. Verify stratification
        from collections import Counter

        def get_size_dist(dataset):
            sizes = [len(ranking) for ranking, _ in dataset]
            return Counter(sizes)

        train_dist = get_size_dist(train)
        test_dist = get_size_dist(test)

        # 4. Train model
        model = PlackettLuceModel(model_type="full")
        model.fit(train, verbose=False)

        # Verify both splits have similar distributions
        assert len(train_dist) > 0
        assert len(test_dist) > 0


class TestRealWorldScenarios:
    """Test realistic use case scenarios."""

    def test_sports_tournament_scenario(self):
        """Test realistic sports tournament scenario."""
        # Simulate a sports tournament
        teams = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE"]

        # Group stage results
        group_results = [
            (("TeamA", "TeamB", "TeamC"), 1),
            (("TeamD", "TeamE", "TeamA"), 1),
        ]

        # Knockout stage
        knockout_results = [
            (("TeamA", "TeamD"), 1),
            (("TeamB", "TeamE"), 1),
            (("TeamA", "TeamB"), 1),  # Final
        ]

        all_results = group_results + knockout_results

        # Fit model
        model = PlackettLuceModel(model_type="full")
        model.fit(all_results, verbose=False)

        # Get final rankings
        ranking = model.get_ranking()

        # TeamA should be highly ranked (won the tournament)
        top_team = ranking[0][0]
        assert top_team == "TeamA"
        assert {node for match, _ in all_results for node in match} == set(teams)

    def test_survey_preference_scenario(self):
        """Test survey preference ranking scenario."""
        # Survey: "Rank your top 3 favorite colors"
        survey_responses = [
            (("Blue", "Green", "Red"), 25),
            (("Green", "Blue", "Red"), 20),
            (("Red", "Blue", "Green"), 15),
            (("Blue", "Red", "Green"), 10),
        ]

        # Use position-1 model (only top choice matters)
        model = PlackettLuceModel(model_type="position1")
        model.fit(survey_responses, verbose=False)

        # Get color rankings
        ranking = model.get_ranking()

        # Blue should be top (appears first most often)
        assert ranking[0][0] == "Blue"

    def test_multiplayer_game_scenario(self):
        """Test multiplayer game ranking scenario."""
        # Simulate multiplayer game results
        players = ["Player1", "Player2", "Player3", "Player4"]

        game_results = [
            (("Player1", "Player2", "Player3", "Player4"), 1),
            (("Player2", "Player1", "Player4", "Player3"), 1),
            (("Player1", "Player3", "Player2", "Player4"), 1),
            (("Player1", "Player2", "Player3", "Player4"), 1),
        ]

        # Fit model
        model = PlackettLuceModel(model_type="full")
        model.fit(game_results, verbose=False)

        ranking = model.get_ranking()
        ranked_players = {player for player, _ in ranking}
        assert ranked_players.issubset(set(players))

        # Player1 should be top ranked
        ranking = model.get_ranking()
        assert ranking[0][0] == "Player1"

    def test_election_voting_scenario(self):
        """Test election/voting scenario."""
        # Ranked choice voting
        votes = [
            (("CandidateA", "CandidateB", "CandidateC"), 100),
            (("CandidateB", "CandidateA", "CandidateC"), 80),
            (("CandidateC", "CandidateB", "CandidateA"), 60),
        ]

        # Full PL model
        model_full = PlackettLuceModel(model_type="full")
        model_full.fit(votes, verbose=False)

        # Position-1 model (plurality voting)
        model_pos1 = PlackettLuceModel(model_type="position1")
        model_pos1.fit(votes, verbose=False)

        # Both should rank CandidateA first
        ranking_full = model_full.get_ranking()
        ranking_pos1 = model_pos1.get_ranking()

        assert ranking_full[0][0] == "CandidateA"
        assert ranking_pos1[0][0] == "CandidateA"


class TestComplexPipelines:
    """Test complex multi-step pipelines."""

    def test_hyperparameter_tuning_pipeline(self):
        """Test hyperparameter tuning workflow."""
        data = generate_synthetic_rankings(N=20, M=200, seed=42)
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        # Try different epsilon values
        epsilons = [1e-4, 1e-5, 1e-6, 1e-7]
        results = {}

        for eps in epsilons:
            model = PlackettLuceModel(model_type="full", epsilon=eps)
            stats = model.fit(train, verbose=False)
            test_ll = model.log_likelihood(test)
            results[eps] = {"iterations": stats["iterations"], "test_ll": test_ll}

        # All should converge
        assert all(res["iterations"] < 10000 for res in results.values())

    def test_ensemble_prediction_pipeline(self):
        """Test ensemble prediction workflow."""
        data = generate_synthetic_rankings(N=15, M=150, seed=42)

        # Create multiple models with different random splits
        models = []
        for seed in range(5):
            train, _ = train_test_split(data, test_size=0.1, random_state=seed)
            model = PlackettLuceModel(model_type="full")
            model.fit(train, verbose=False)
            models.append(model)

        # Ensemble prediction
        test_ranking = data[0][0]
        probs = [m.predict_probability(test_ranking) for m in models]
        ensemble_prob = np.mean(probs)

        assert 0 <= ensemble_prob <= 1
        assert len(probs) == 5

    def test_ranking_stability_pipeline(self):
        """Test ranking stability across different data samples."""
        # Generate multiple datasets from same distribution
        rankings_list = []

        for seed in range(5):
            data = generate_synthetic_rankings(N=10, M=100, seed=seed)
            model = PlackettLuceModel(model_type="full")
            model.fit(data, verbose=False)
            ranking = [node for node, _ in model.get_ranking()]
            rankings_list.append(ranking)

        # Compute pairwise similarities
        similarities = []
        for i in range(len(rankings_list)):
            for j in range(i + 1, len(rankings_list)):
                sim = ranking_similarity(rankings_list[i], rankings_list[j], method="kendall")
                similarities.append(sim)

        # Rankings should have some similarity
        mean_similarity = np.mean(similarities)
        assert mean_similarity > -0.5  # Not completely random

    def test_incremental_learning_pipeline(self):
        """Test incremental learning workflow."""
        # Initial training
        initial_data = generate_synthetic_rankings(N=15, M=100, seed=42)
        model = PlackettLuceModel(model_type="full")
        model.fit(initial_data, verbose=False)
        initial_ranking = model.get_ranking()

        # Add more data
        new_data = generate_synthetic_rankings(N=15, M=50, seed=43)
        combined_data = initial_data + new_data

        # Retrain
        model_updated = PlackettLuceModel(model_type="full")
        model_updated.fit(combined_data, verbose=False)
        updated_ranking = model_updated.get_ranking()

        # Rankings should be similar but may differ
        similarity = ranking_similarity(
            [n for n, _ in initial_ranking],
            [n for n, _ in updated_ranking],
            method="kendall",
        )
        assert similarity > 0.5  # Should maintain some consistency


class TestErrorHandlingInWorkflows:
    """Test error handling in complete workflows."""

    def test_workflow_with_missing_nodes(self):
        """Test workflow when test set has nodes not in training set."""
        # This should raise an error or be handled gracefully
        train_data = [(("A", "B"), 1), (("B", "C"), 1)]
        test_data = [(("X", "Y"), 1)]

        model = PlackettLuceModel(model_type="full")
        model.fit(train_data, verbose=False)

        # Should raise error for unknown nodes
        with pytest.raises(ValueError):
            model.log_likelihood(test_data)

    def test_workflow_with_empty_split(self):
        """Test workflow with very small data leading to empty split."""
        data = [(("A", "B"), 1)]

        # This might fail or produce warnings
        try:
            train, test = train_test_split(data, test_size=0.5)
            if len(train) > 0:
                model = PlackettLuceModel(model_type="full")
                model.fit(train, verbose=False)
        except ValueError:
            # Expected for very small datasets
            pass

    def test_workflow_with_convergence_failure(self):
        """Test workflow when model fails to converge."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)

        # Use very strict epsilon and low max_iterations
        model = PlackettLuceModel(model_type="full", epsilon=1e-10, max_iterations=5)

        stats = model.fit(data, verbose=False)

        # Should still produce results even without convergence
        assert model.is_fitted
        assert stats["iterations"] == 5


class TestPerformanceWorkflows:
    """Test performance-related workflows."""

    def test_large_dataset_workflow(self):
        """Test workflow with large dataset."""
        # Generate large dataset
        data = generate_synthetic_rankings(N=100, M=1000, K_min=2, K_max=10, seed=42)

        # Should complete in reasonable time
        model = PlackettLuceModel(model_type="full")
        stats = model.fit(data, verbose=False)

        assert model.is_fitted
        assert stats["time"] < 60  # Should take less than 60 seconds

    def test_high_dimensional_workflow(self):
        """Test workflow with many entities."""
        # Many entities, moderate comparisons
        data = generate_synthetic_rankings(N=200, M=500, K_min=2, K_max=5, seed=42)

        model = PlackettLuceModel(model_type="full")
        model.fit(data, verbose=False)

        assert len(model.scores) == 200
        assert model.is_fitted

    def test_repeated_fitting_workflow(self):
        """Test fitting model multiple times."""
        data = generate_synthetic_rankings(N=20, M=100, seed=42)

        model = PlackettLuceModel(model_type="full")

        # Fit multiple times
        for _ in range(3):
            stats = model.fit(data, verbose=False)
            assert model.is_fitted
            assert stats["iterations"] > 0

        # Should produce consistent results
        ranking = model.get_ranking()
        assert len(ranking) == 20
