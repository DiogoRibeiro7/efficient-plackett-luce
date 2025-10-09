# Efficient Plackett-Luce Documentation

Welcome to the documentation for **efficient-plackett-luce**, a high-performance Python package for ranking inference from multi-body comparisons.

## What is Plackett-Luce?

The Plackett-Luce (PL) model is a probabilistic framework for learning rankings from comparison data. Given observations of how entities (teams, products, players) perform relative to each other, the PL model infers underlying skill/quality scores.

### Key Features

- **Multi-body comparisons**: Unlike Bradley-Terry (pairwise only), PL handles comparisons with 3+ entities
- **Probabilistic**: Provides uncertainty estimates via probability distributions
- **Fast inference**: This implementation uses Newman's efficient algorithm (5-70x speedup)
- **Flexible**: Works with tournaments, surveys, elections, and more

## Quick Start

```python
from plackett_luce import PlackettLuceModel

# Define your comparison data
data = [
    (("TeamA", "TeamB", "TeamC"), 1),  # TeamA > TeamB > TeamC
    (("TeamB", "TeamA"), 2),            # TeamB > TeamA (twice)
]

# Fit the model
model = PlackettLuceModel(model_type='full')
model.fit(data)

# Get rankings
rankings = model.get_ranking()
for rank, (entity, score) in enumerate(rankings, 1):
    print(f"{rank}. {entity}: {score:.4f}")
```

## Installation

```bash
pip install efficient-plackett-luce
```

## When to Use This Package

### ✅ Good Use Cases

- **Sports tournaments** with group stages and knockout rounds
- **Multiplayer games** where 3+ players compete simultaneously
- **Elections and surveys** where voters rank multiple candidates
- **Product rankings** from comparison data
- **Academic authorship** analysis (author order reflects contribution)

### ❌ Not Ideal For

- Simple pairwise comparisons only (use Bradley-Terry instead)
- Very small datasets (< 20 comparisons)
- Cases where you need deep learning integration

## Core Concepts

### 1. Hyperedge Format

Data consists of `(ranking, weight)` tuples:

```python
hyperedges = [
    (("Gold", "Silver", "Bronze"), 1),    # One occurrence
    (("Player1", "Player2"), 5),          # Five occurrences
]
```

### 2. Model Types

#### Full Plackett-Luce
Uses all ranking information. Best when all positions matter.

```python
model = PlackettLuceModel(model_type='full')
```

**Probability formula:**
$P(\omega | \pi) = \prod_{r=1}^{K-1} \frac{\pi_{\omega_r}}{\sum_{q=r}^{K} \pi_{\omega_q}}$

#### Position-1-Breaking
Only uses who came first. Best when only winners matter.

```python
model = PlackettLuceModel(model_type='position1')
```

**Probability formula:**
$P(\omega | \pi) = \frac{\pi_{\omega_1}}{\sum_{q=1}^{K} \pi_{\omega_q}}$

### 3. Score Interpretation

Scores $\pi_i$ represent relative strength:
- **Higher score** = stronger/better entity
- **Ratio interpretation**: $\frac{\pi_i}{\pi_j}$ is the odds that entity $i$ beats entity $j$
- **Normalization**: $\prod_{i=1}^{N} \pi_i = 1$

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Examples](examples.md) - Detailed usage examples
- [GitHub Repository](https://github.com/yourusername/efficient-plackett-luce)

## Citation

If you use this package in research, please cite:

```bibtex
@article{yeung2025efficient,
  title={Efficient inference of rankings from multi-body comparisons},
  author={Yeung, Jack and Kaiser, Daniel and Radicchi, Filippo},
  journal={arXiv preprint arXiv:2501.16565},
  year={2025}
}
```
