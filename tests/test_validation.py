"""
Tests for validation utilities.
"""

import pytest
import numpy as np
from plackett_luce.validation import (
    train_test_split,
    stratified_split_by_size,
    temporal_split,
    cross_validate
)
from plackett_luce.utils import generate_synthetic_rankings

class TestValidationFunctions:
    """Test suite for validation utilities."""
    
    def test_train_test_split(self):
        """Test basic train/test split."""
        data = generate_synthetic_rankings(N=10, M=100, seed=42)
        
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        
        assert len(train) + len(test) == len(data)
        assert len(test) == 20
        assert len(train) == 80
    
    def test_train_test_split_sizes(self):
        """Test different test sizes."""
        data = generate_synthetic_rankings(N=10, M=100, seed=42)
        
        for test_size in [0.1, 0.3, 0.5]:
            train, test = train_test_split(data, test_size=test_size, random_state=42)
            actual_ratio = len(test) / len(data)
            assert np.isclose(actual_ratio, test_size, atol=0.05)
    
    def test_train_test_split_no_overlap(self):
        """Test that train and test sets don't overlap."""
        data = [
            (("A", "B"), 1),
            (("C", "D"), 1),
            (("E", "F"), 1),
            (("G", "H"), 1),
        ]
        
        train, test = train_test_split(data, test_size=0.5, random_state=42)
        
        # Convert to sets for comparison
        train_set = set(str(x) for x in train)
        test_set = set(str(x) for x in test)
        
        assert len(train_set & test_set) == 0
    
    def test_train_test_split_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        data = generate_synthetic_rankings(N=10, M=100, seed=42)
        
        train1, test1 = train_test_split(data, test_size=0.2, random_state=42)
        train2, test2 = train_test_split(data, test_size=0.2, random_state=42)
        
        assert train1 == train2
        assert test1 == test2
    
    def test_train_test_split_invalid_size(self):
        """Test that invalid test_size raises error."""
        data = [(("A", "B"), 1)]
        
        with pytest.raises(ValueError):
            train_test_split(data, test_size=0.0)
        
        with pytest.raises(ValueError):
            train_test_split(data, test_size=1.0)
        
        with pytest.raises(ValueError):
            train_test_split(data, test_size=-0.1)
    
    def test_stratified_split(self):
        """Test stratified splitting maintains size distribution."""
        data = [
            (("A", "B"), 1),
            (("C", "D"), 1),
            (("E", "F", "G"), 1),
            (("H", "I", "J"), 1),
        ]
        
        train, test = stratified_split_by_size(data, test_size=0.5, random_state=42)
        
        # Should have one size-2 and one size-3 in each split
        train_sizes = [len(r) for r, _ in train]
        test_sizes = [len(r) for r, _ in test]
        
        assert 2 in train_sizes and 3 in train_sizes
        assert 2 in test_sizes and 3 in test_sizes
    
    def test_temporal_split(self):
        """Test temporal splitting (no shuffle)."""
        data = [
            (("A", "B"), i) for i in range(10)
        ]
        
        train, test = temporal_split(data, test_size=0.2)
        
        assert len(train) == 8
        assert len(test) == 2
        assert train == data[:8]
        assert test == data[8:]
    
    def test_cross_validate(self):
        """Test k-fold cross-validation."""
        data = generate_synthetic_rankings(N=20, M=100, seed=42)
        
        results = cross_validate(
            data,
            model_type='full',
            n_splits=3,
            compare_projected=True,
            random_state=42,
            verbose=False
        )
        
        assert 'pl_mean' in results
        assert 'pl_std' in results
        assert 'pl_scores' in results
        assert len(results['pl_scores']) == 3
        
        assert 'projected_mean' in results
        assert 'winner' in results
        assert results['winner'] in ['PL', 'Projected']
    
    def test_cross_validate_without_projection(self):
        """Test CV without projection comparison."""
        data = generate_synthetic_rankings(N=20, M=100, seed=42)
        
        results = cross_validate(
            data,
            model_type='full',
            n_splits=3,
            compare_projected=False,
            random_state=42,
            verbose=False
        )
        
        assert 'pl_mean' in results
        assert 'projected_mean' not in results
        assert 'winner' not in results
