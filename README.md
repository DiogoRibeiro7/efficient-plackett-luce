# Efficient Plackett-Luce

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Fast inference for multi-body rankings using Newman's efficient algorithm**

This package implements the Plackett-Luce model for ranking inference from multi-body comparisons, achieving **5-70x speedup** over traditional methods by using Newman's efficient iterative algorithm with Numba JIT compilation.

## 📚 Overview

The Plackett-Luce (PL) model is a probabilistic framework for learning rankings from comparison data. While most ranking methods focus on pairwise comparisons, many real-world scenarios involve **multi-body comparisons**:

- 🏆 **Sports tournaments** - Group stages, multiple teams per match
- 🎮 **Multiplayer games** - Rankings from games with 3+ players
- 🗳️ **Elections & surveys** - Voters rank multiple candidates
- 📝 **Academic authorship** - Author order reflects contribution
- ⭐ **Product rankings** - Multiple items competing simultaneously

This implementation is based on the paper:

> **"Efficient inference of rankings from multi-body comparisons"**<br>
> Jack Yeung, Daniel Kaiser, and Filippo Radicchi<br>
> _arXiv:2501.16565_, 2025

## ✨ Key Features

- ⚡ **Fast**: Newman's algorithm with Numba JIT compilation (5-70x faster)
- 🎯 **Complete**: Full PL model + Position-1-breaking variant + Projected models
- 🔬 **Validated**: Cross-validation, train/test splits, model selection
- 💾 **Production-ready**: Model persistence, input validation, error handling
- 📊 **Flexible**: Works with any comparison size (K ≥ 2)
- 🐍 **Pure Python**: No C/Fortran dependencies (just Numba)

## 🚀 Quick Start

### Installation

```bash
pip install efficient-plackett-luce
```

Or install from source:

```bash
git clone https://github.com/yourusername/efficient-plackett-luce.git
cd efficient-plackett-luce
pip install -e .
```

### Basic Usage

```python
from plackett_luce import PlackettLuceModel

# Define tournament results: (ranking, weight) pairs
# Ranking is a tuple where earlier positions are better
results = [
    (("TeamA", "TeamB", "TeamC"), 1),      # TeamA > TeamB > TeamC
    (("TeamB", "TeamA"), 2),                # TeamB > TeamA (occurred twice)
    (("TeamC", "TeamA", "TeamB"), 1),       # TeamC > TeamA > TeamB
]

# Fit the model
model = PlackettLuceModel(model_type='full')
model.fit(results, verbose=True)

# Get rankings
for rank, (team, score) in enumerate(model.get_ranking(), 1):
    print(f"{rank}. {team}: {score:.4f}")

# Predict probability of a specific ranking
prob = model.predict_probability(("TeamA", "TeamB", "TeamC"))
print(f"P(TeamA > TeamB > TeamC) = {prob:.6f}")

# Save model
model.save("tournament_model.pkl")

# Load model later
loaded_model = PlackettLuceModel.load("tournament_model.pkl")
```

## 📖 Documentation

### Model Types

#### 1\. Full Plackett-Luce Model

Uses complete ranking information. Best for scenarios where all positions matter.

```python
model = PlackettLuceModel(model_type='full')
```

#### 2\. Position-1-Breaking Model

Only uses who came first. Best when only the winner matters.

```python
model = PlackettLuceModel(model_type='position1')
```

#### 3\. Projected Pairwise Model

Converts multi-body to pairwise comparisons for comparison purposes.

```python
from plackett_luce import ProjectedPlackettLuce
model = ProjectedPlackettLuce(model_type='full')
```

### Cross-Validation

```python
from plackett_luce import cross_validate

# Compare full PL vs projected pairwise model
results = cross_validate(
    data, 
    model_type='full',
    n_splits=5,
    compare_projected=True,
    verbose=True
)

print(f"PL Model: {results['pl_mean']:.4f} ± {results['pl_std']:.4f}")
print(f"Projected: {results['projected_mean']:.4f} ± {results['projected_std']:.4f}")
print(f"Winner: {results['winner']} (improvement: {results['improvement']:.2f}%)")
```

### Train/Test Split

```python
from plackett_luce.validation import train_test_split, stratified_split_by_size

# Standard random split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Stratified by comparison size (maintains K distribution)
train, test = stratified_split_by_size(data, test_size=0.2, random_state=42)

# Temporal split (for time-series data)
from plackett_luce.validation import temporal_split
train, test = temporal_split(tournament_rounds, test_size=0.2)
```

### Synthetic Data Generation

```python
from plackett_luce.utils import generate_synthetic_rankings

# Generate synthetic tournament data
data = generate_synthetic_rankings(
    N=100,        # 100 entities
    M=1000,       # 1000 comparisons
    K_min=2,      # Min 2 entities per comparison
    K_max=10,     # Max 10 entities per comparison
    seed=42
)
```

## 📊 Examples

### Example 1: FIFA World Cup Rankings

```python
from plackett_luce import PlackettLuceModel

# Historical World Cup results
wc_results = [
    (("Brazil", "Germany", "Italy", "Argentina"), 1),  # Group stage
    (("France", "Belgium", "England", "Croatia"), 1),
    (("Brazil", "Germany"), 3),                         # Final (3 times)
    (("Argentina", "Netherlands"), 2),                  # Semi-final
]

model = PlackettLuceModel(model_type='full')
model.fit(wc_results, verbose=True)

print("\nTop 10 Teams:")
for rank, (team, score) in enumerate(model.get_ranking(top_k=10), 1):
    print(f"{rank:2d}. {team:15s} {score:.4f}")
```

### Example 2: Multiplayer Game Rankings

```python
# Poker tournament results
poker_results = [
    (("Alice", "Bob", "Charlie", "Diana"), 1),
    (("Bob", "Alice", "Charlie"), 1),
    (("Charlie", "Diana", "Alice", "Bob"), 1),
]

model = PlackettLuceModel(model_type='full')
model.fit(poker_results)

# Predict outcome of a specific seating
prob = model.predict_probability(("Alice", "Bob", "Charlie"))
print(f"Probability Alice wins: {prob:.3f}")
```

### Example 3: Survey Preference Analysis

```python
# Survey: rank your top 3 favorite movies
survey_results = [
    (("Inception", "Interstellar", "The Matrix"), 5),    # 5 people chose this
    (("The Matrix", "Inception", "Interstellar"), 3),
    (("Interstellar", "The Matrix", "Inception"), 2),
]

# Use position-1-breaking (only top choice matters)
model = PlackettLuceModel(model_type='position1')
model.fit(survey_results)

print("Movie Rankings:")
for rank, (movie, score) in enumerate(model.get_ranking(), 1):
    print(f"{rank}. {movie}")
```

### Example 4: Model Comparison

```python
from plackett_luce import PlackettLuceModel, ProjectedPlackettLuce
from plackett_luce.validation import train_test_split

# Split data
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Compare full PL vs projected pairwise
models = {
    'Full PL': PlackettLuceModel(model_type='full'),
    'Projected PL': ProjectedPlackettLuce(model_type='full'),
    'Position-1': PlackettLuceModel(model_type='position1'),
}

for name, model in models.items():
    model.fit(train)
    ll = model.log_likelihood(test)
    print(f"{name:15s} Log-likelihood: {ll:.4f}")
```

## 🔧 API Reference

### PlackettLuceModel

```python
model = PlackettLuceModel(
    model_type='full',           # 'full' or 'position1'
    epsilon=1e-6,                 # Convergence threshold
    max_iterations=10000          # Max iterations
)
```

**Methods:**

- `fit(hyperedges, verbose=False)` - Fit model to data
- `predict_probability(ranking)` - Compute P(ranking)
- `log_likelihood(hyperedges)` - Compute log-likelihood
- `get_ranking(top_k=None)` - Get ranked list of entities
- `get_score(node_id)` - Get score for specific entity
- `save(filepath)` - Save model to disk
- `load(filepath)` - Load model from disk (class method)

### Input Format

Data should be a list of `(ranking, weight)` tuples:

```python
hyperedges = [
    (("entity1", "entity2", "entity3"), 1),  # entity1 > entity2 > entity3
    (("entity2", "entity1"), 2),              # Occurred twice
]
```

- **ranking**: Tuple of entity IDs in decreasing order of performance
- **weight**: Integer count of how many times this ranking occurred

## ⚡ Performance

Benchmark on synthetic data (N=1000 nodes, M=10000 comparisons):

Algorithm          | Iterations | Time  | Speedup
------------------ | ---------- | ----- | -------
Zermelo (baseline) | 169        | 4.2s  | 1x
Newman (ours)      | 11         | 0.28s | **15x**

The speedup increases with:

- ✅ Larger datasets
- ✅ More complex comparisons (larger K)
- ✅ Denser hypergraphs

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=plackett_luce tests/

# Run specific test
pytest tests/test_core.py::test_fit_convergence
```

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@article{yeung2025efficient,
  title={Efficient inference of rankings from multi-body comparisons},
  author={Yeung, Jack and Kaiser, Daniel and Radicchi, Filippo},
  journal={arXiv preprint arXiv:2501.16565},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:

- ✅ Code follows Black formatting
- ✅ Tests pass (`pytest`)
- ✅ New features have tests
- ✅ Documentation is updated

## 📜 License

This project is licensed under the MIT License - see the <LICENSE> file for details.

## 🙏 Acknowledgments

- **Newman's Algorithm**: Based on M. Newman's efficient Bradley-Terry algorithm
- **Original Paper**: Yeung, Kaiser, and Radicchi (2025)
- **Numba**: For JIT compilation enabling high performance in pure Python

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/diogoribeiro7/efficient-plackett-luce/issues)
- **Email**: <dfr@esmad.ipp.pt>
- **Paper**: [arXiv:2501.16565](https://arxiv.org/abs/2501.16565)

## 🗺️ Roadmap

- [ ] GPU acceleration with CuPy
- [ ] Sparse matrix optimization for very large graphs
- [ ] Time-varying Plackett-Luce models
- [ ] Bayesian uncertainty quantification
- [ ] Integration with scikit-learn pipelines
- [ ] Web API for ranking as a service

--------------------------------------------------------------------------------

**Made with ❤️ for the ranking community**
