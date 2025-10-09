"""
Tests for core Plackett-Luce model functionality.
"""

import pytest
import numpy as np
from plackett_luce import PlackettLuceModel
from plackett_luce.utils import generate_synthetic_rankings

class TestPlackettLuceModel:
    """Test suite for PlackettLuceModel class."""
    
    def test_initialization(self):
        """Test model initialization with different parameters."""
        model = PlackettLuceModel(model_type='full')
        assert model.model_type == 'full'
        assert model.epsilon == 1e-6
        assert model.is_fitted == False
        
        model_pos1 = PlackettLuceModel(model_type='position1', epsilon=1e-4)
        assert model_pos1.model_type == 'position1'
        assert model_pos1.epsilon == 1e-4
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            PlackettLuceModel(model_type='invalid')
    
    def test_fit_simple_data(self):
        """Test fitting on simple data."""
        data = [
            (("A", "B", "C"), 1),
            (("B", "C", "A"), 1),
            (("A", "B"), 2),
        ]
        
        model = PlackettLuceModel(model_type='full')
        stats = model.fit(data, verbose=False)
        
        assert model.is_fitted == True
        assert 'iterations' in stats
        assert 'time' in stats
        assert stats['iterations'] > 0
    
    def test_fit_convergence(self):
        """Test that model converges."""
        data = generate_synthetic_rankings(N=20, M=100, K_min=2, K_max=5, seed=42)
        
        model = PlackettLuceModel(model_type='full')
        stats = model.fit(data, verbose=False)
        
        assert stats['iterations'] < model.max_iterations
        assert model.scores is not None
        assert len(model.scores) == 20
    
    def test_get_ranking(self):
        """Test getting rankings."""
        data = [(("A", "B", "C"), 2), (("A", "C"), 1)]
        
        model = PlackettLuceModel(model_type='full')
        model.fit(data, verbose=False)
        
        ranking = model.get_ranking()
        
        assert len(ranking) == 3
        assert ranking[0][0] == "A"  # A should be top
        assert all(isinstance(score, float) for _, score in ranking)
    
    def test_get_ranking_top_k(self):
        """Test getting top-k rankings."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)
        
        model = PlackettLuceModel(model_type='full')
        model.fit(data, verbose=False)
        
        top_3 = model.get_ranking(top_k=3)
        
        assert len(top_3) == 3
        assert top_3[0][1] >= top_3[1][1] >= top_3[2][1]
    
    def test_predict_probability(self):
        """Test probability prediction."""
        data = [(("A", "B", "C"), 1), (("A", "B"), 1)]
        
        model = PlackettLuceModel(model_type='full')
        model.fit(data, verbose=False)
        
        prob = model.predict_probability(("A", "B", "C"))
        
        assert 0 <= prob <= 1
        assert isinstance(prob, float)
    
    def test_predict_probability_unknown_node(self):
        """Test that predicting with unknown node raises error."""
        data = [(("A", "B"), 1)]
        
        model = PlackettLuceModel(model_type='full')
        model.fit(data, verbose=False)
        
        with pytest.raises(ValueError):
            model.predict_probability(("A", "Z"))
    
    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)
        
        model = PlackettLuceModel(model_type='full')
        model.fit(data, verbose=False)
        
        ll = model.log_likelihood(data)
        
        assert isinstance(ll, float)
        assert ll < 0  # Log-likelihood should be negative
    
    def test_save_load(self, tmp_path):
        """Test saving and loading models."""
        data = [(("A", "B", "C"), 1), (("B", "C"), 1)]
        
        model = PlackettLuceModel(model_type='full')
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
    
    def test_position1_model(self):
        """Test position-1-breaking model."""
        data = [(("A", "B", "C"), 2), (("B", "A", "C"), 1)]
        
        model = PlackettLuceModel(model_type='position1')
        stats = model.fit(data, verbose=False)
        
        assert model.is_fitted == True
        assert stats['iterations'] > 0
        
        prob = model.predict_probability(("A", "B", "C"))
        assert 0 <= prob <= 1
    
    def test_empty_data(self):
        """Test that empty data raises error."""
        model = PlackettLuceModel(model_type='full')
        
        with pytest.raises(ValueError):
            model.fit([], verbose=False)
    
    def test_invalid_data_format(self):
        """Test that invalid data format raises error."""
        model = PlackettLuceModel(model_type='full')
        
        with pytest.raises(ValueError):
            model.fit([("A", "B")], verbose=False)  # Missing weight
    
    def test_score_normalization(self):
        """Test that scores are properly normalized."""
        data = generate_synthetic_rankings(N=10, M=50, seed=42)
        
        model = PlackettLuceModel(model_type='full')
        model.fit(data, verbose=False)
        
        # Geometric mean should be close to 1
        geometric_mean = np.exp(np.mean(np.log(model.scores)))
        assert np.isclose(geometric_mean, 1.0, atol=1e-6)
