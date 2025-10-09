# Examples

Detailed examples demonstrating various use cases.

## Table of Contents

1. [Basic Ranking](#basic-ranking)
2. [Sports Tournaments](#sports-tournaments)
3. [Survey Analysis](#survey-analysis)
4. [Model Selection](#model-selection)
5. [Cross-Validation](#cross-validation)
6. [Custom Applications](#custom-applications)

---

## Basic Ranking

### Simple Tournament

```python
from plackett_luce import PlackettLuceModel

# Define tournament results
results = [
    (("Alice", "Bob", "Charlie"), 1),
    (("Bob", "Charlie", "Alice"), 1),
    (("Alice", "Bob"), 2),  # Occurred twice
]

# Fit model
model = PlackettLuceModel(model_type='full')
model.fit(results, verbose=True)

# Get rankings
print("\nPlayer Rankings:")
for rank, (player, score) in enumerate(model.get_ranking(), 1):
    print(f"{rank}. {player:10s} {score:.4f}")

# Make predictions
prob = model.predict_probability(("Alice", "Bob", "Charlie"))
print(f"\nP(Alice > Bob > Charlie) = {prob:.6f}")
```

### Weighted Comparisons

```python
# Some outcomes occurred multiple times
data = [
    (("TeamA", "TeamB"), 5),    # TeamA beat TeamB 5 times
    (("TeamB", "TeamA"), 2),    # TeamB beat TeamA 2 times
    (("TeamA", "TeamC"), 3),
]

model = PlackettLuceModel(model_type='full')
model.fit(data)

# TeamA should be ranked highest
rankings = model.get_ranking()
assert rankings[0][0] == "TeamA"
```

---

## Sports Tournaments

### FIFA World Cup Style

```python
from plackett_luce import PlackettLuceModel

# Group stage results
group_a = [("Brazil", "Croatia", "Mexico", "Cameroon"), 1]
group_b = [("Spain", "Netherlands", "Chile", "Australia"), 1]

# Knockout results
knockout = [
    (("Brazil", "Chile"), 1),
    (("Germany", "France"), 1),
    (("Germany", "Brazil"), 1),    # Semi-final
    (("Germany", "Argentina"), 1),  # Final
]

all_matches = group_a + group_b + knockout

model = PlackettLuceModel(model_type='full')
model.fit(all_matches)

print("Top 5 Teams:")
for rank, (team, score) in enumerate(model.get_ranking(top_k=5), 1):
    print(f"{rank}. {team}")
```

### League Standings

```python
# Regular season results
results = []

# Each team plays every other team twice
teams = ["TeamA", "TeamB", "TeamC", "TeamD"]

for home in teams:
    for away in teams:
        if home != away:
            # Add match results (home team advantage)
            results.append(((home, away), 1))

model = PlackettLuceModel(model_type='full')
model.fit(results)

# Generate league table
print("\nLeague Standings:")
print(f"{'Pos':<4} {'Team':<10} {'Score':<8}")
print("-" * 25)
for rank, (team, score) in enumerate(model.get_ranking(), 1):
    print(f"{rank:<4} {team:<10} {score:<8.4f}")
```

---

## Survey Analysis

### Preference Ranking Survey

```python
# Survey: "Rank your top 3 favorite fruits"
survey_responses = [
    (("Apple", "Banana", "Orange"), 15),      # 15 people
    (("Banana", "Apple", "Orange"), 10),
    (("Orange", "Apple", "Banana"), 8),
    (("Banana", "Orange", "Apple"), 5),
]

model = PlackettLuceModel(model_type='full')
model.fit(survey_responses)

print("Fruit Popularity Ranking:")
for rank, (fruit, score) in enumerate(model.get_ranking(), 1):
    print(f"{rank}. {fruit}: {score:.4f}")
```

### Election Data (Position-1 Only)

```python
# Election: voters only care about their first choice
votes = [
    (("Candidate_A", "Candidate_B", "Candidate_C"), 120),
    (("Candidate_B", "Candidate_A", "Candidate_C"), 95),
    (("Candidate_C", "Candidate_B", "Candidate_A"), 80),
]

# Use position-1-breaking model
model = PlackettLuceModel(model_type='position1')
model.fit(votes)

print("Election Results:")
for rank, (candidate, score) in enumerate(model.get_ranking(), 1):
    pct = score / sum(s for _, s in model.get_ranking()) * 100
    print(f"{rank}. {candidate}: {pct:.1f}%")
```

---

## Model Selection

### Comparing Full vs Position-1

```python
from plackett_luce import PlackettLuceModel
from plackett_luce.validation import train_test_split

# Generate or load your data
data = [...]  # Your comparison data

# Split data
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Test different model types
models = {
    'Full PL': PlackettLuceModel(model_type='full'),
    'Position-1': PlackettLuceModel(model_type='position1'),
}

results = {}
for name, model in models.items():
    model.fit(train)
    ll = model.log_likelihood(test)
    results[name] = ll
    print(f"{name:15s} Test LL: {ll:.4f}")

# Select best model
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")
```

### Full PL vs Projected Pairwise

```python
from plackett_luce import PlackettLuceModel, ProjectedPlackettLuce

# Fit both models
model_pl = PlackettLuceModel(model_type='full')
model_proj = ProjectedPlackettLuce(model_type='full')

model_pl.fit(train)
model_proj.fit(train)

# Compare on test set
ll_pl = model_pl.log_likelihood(test)
ll_proj = model_proj.log_likelihood(test)

print(f"Full PL:     {ll_pl:.4f}")
print(f"Projected:   {ll_proj:.4f}")
print(f"Difference:  {ll_pl - ll_proj:.4f}")

if ll_pl > ll_proj:
    print("\n✓ Multi-body structure is meaningful")
else:
    print("\n✗ Pairwise projection is sufficient")
```

---

## Cross-Validation

### K-Fold Cross-Validation

```python
from plackett_luce import cross_validate

# Perform 5-fold CV
results = cross_validate(
    data,
    model_type='full',
    n_splits=5,
    compare_projected=True,
    random_state=42,
    verbose=True
)

print("\nCross-Validation Results:")
print(f"Full PL:   {results['pl_mean']:.4f} ± {results['pl_std']:.4f}")
print(f"Projected: {results['projected_mean']:.4f} ± {results['projected_std']:.4f}")
print(f"\nWinner: {results['winner']}")
print(f"Improvement: {results['improvement']:.2f}%")
```

### Learning Curves

```python
from plackett_luce import PlackettLuceModel
from plackett_luce.validation import train_test_split
import numpy as np

# Vary training set size
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
test_scores = []

for train_size in train_sizes:
    train, test = train_test_split(data, test_size=1-train_size, random_state=42)
    
    model = PlackettLuceModel(model_type='full')
    model.fit(train, verbose=False)
    
    test_ll = model.log_likelihood(test)
    test_scores.append((len(train), test_ll))

print("Learning Curve:")
print(f"{'Train Size':<12} {'Test LL':<10}")
for n, score in test_scores:
    print(f"{n:<12} {score:>8.4f}")
```

---

## Custom Applications

### Author Contribution Analysis

```python
# Scientific papers with author lists
# (last author is typically the senior/corresponding author)
papers = [
    (("Senior_A", "Postdoc_X", "Student_Y"), 1),
    (("Senior_A", "Student_Z"), 1),
    (("Senior_B", "Postdoc_X", "Senior_A", "Student_W"), 1),
]

# Reverse order so senior authors appear first
papers_reversed = [
    (tuple(reversed(authors)), weight) 
    for authors, weight in papers
]

model = PlackettLuceModel(model_type='position1')
model.fit(papers_reversed)

print("Author Leadership Ranking:")
for rank, (author, score) in enumerate(model.get_ranking(), 1):
    print(f"{rank}. {author}: {score:.4f}")
```

### Product Ranking from A/B/C Tests

```python
# Multi-variant testing results
# (product shown, products clicked in order)
test_results = [
    (("ProductA", "ProductB", "ProductC"), 50),    # 50 users
    (("ProductB", "ProductA", "ProductC"), 30),
    (("ProductC", "ProductA", "ProductB"), 20),
]

model = PlackettLuceModel(model_type='position1')  # Only care about clicks
model.fit(test_results)

print("Product Click-Through Ranking:")
for rank, (product, score) in enumerate(model.get_ranking(), 1):
    ctr = score / sum(s for _, s in model.get_ranking()) * 100
    print(f"{rank}. {product}: {ctr:.1f}% relative CTR")
```

### Gaming Leaderboard

```python
# Multiplayer game results
matches = [
    (("Player1", "Player2", "Player3", "Player4"), 1),
    (("Player2", "Player1", "Player4", "Player3"), 1),
    (("Player1", "Player3", "Player2", "Player4"), 1),
    # ... more matches
]

model = PlackettLuceModel(model_type='full')
model.fit(matches)

print("=== LEADERBOARD ===")
for rank, (player, score) in enumerate(model.get_ranking(), 1):
    print(f"#{rank:2d} {player:15s} Rating: {score:.2f}")

# Predict match outcome
prob = model.predict_probability(("Player1", "Player2", "Player3"))
print(f"\nP(Player1 wins next match) ≈ {prob:.1%}")
```

---

## Advanced Topics

### Time-Varying Rankings

```python
from plackett_luce.validation import temporal_split

# Tournament data over time (chronologically ordered)
all_matches = load_tournament_history()

# Train on past, test on recent
train, test = temporal_split(all_matches, test_size=0.1)

model = PlackettLuceModel(model_type='full')
model.fit(train)

# Evaluate predictive power on recent matches
recent_ll = model.log_likelihood(test)
print(f"Predictive log-likelihood: {recent_ll:.4f}")
```

### Ensemble Predictions

```python
# Train multiple models and ensemble predictions
models = []
n_models = 5

for seed in range(n_models):
    train, _ = train_test_split(data, test_size=0.1, random_state=seed)
    model = PlackettLuceModel(model_type='full')
    model.fit(train, verbose=False)
    models.append(model)

# Ensemble prediction (average probabilities)
test_ranking = ("TeamA", "TeamB", "TeamC")
probs = [m.predict_probability(test_ranking) for m in models]
ensemble_prob = np.mean(probs)

print(f"Individual predictions: {probs}")
print(f"Ensemble prediction: {ensemble_prob:.4f}")
```

---

## Tips and Best Practices

### 1. Data Quality

```python
# Remove duplicates
from collections import Counter

def deduplicate_data(hyperedges):
    """Combine duplicate rankings."""
    counter = Counter()
    for ranking, weight in hyperedges:
        counter[ranking] += weight
    return list(counter.items())

clean_data = deduplicate_data(raw_data)
```

### 2. Handling Ties

```python
# If you have ties, break them randomly or treat as separate comparisons
def break_ties(ranking_with_ties):
    """
    ranking_with_ties: [(entities_at_pos_1,), (entities_at_pos_2,), ...]
    Returns: All possible strict orderings
    """
    from itertools import permutations, product
    
    # Generate all permutations for each tie group
    tie_perms = [list(permutations(group)) for group in ranking_with_ties]
    
    # Cartesian product of all combinations
    all_orderings = []
    for combo in product(*tie_perms):
        # Flatten
        ordering = tuple(item for group in combo for item in group)
        all_orderings.append(ordering)
    
    return all_orderings

# Example: [(A, B), (C,)] means A and B tied for first, C is second
tied_result = [(("TeamA", "TeamB"), ("TeamC",))]
strict_orderings = break_ties(tied_result[0])
# Results in: [("TeamA", "TeamB", "TeamC"), ("TeamB", "TeamA", "TeamC")]
```

### 3. Cold Start Problem

```python
# Add regularization for entities with few observations
def add_virtual_comparisons(data, entities, virtual_weight=0.1):
    """Add small uniform prior."""
    import itertools
    
    # Add weak pairwise comparisons for all pairs
    virtual_data = []
    for e1, e2 in itertools.combinations(entities, 2):
        # Each entity beats each other with small weight
        virtual_data.append(((e1, e2), virtual_weight))
        virtual_data.append(((e2, e1), virtual_weight))
    
    return data + virtual_data

regularized_data = add_virtual_comparisons(data, all_entities)
```

### 4. Interpreting Scores

```python
# Convert scores to win probabilities
def pairwise_win_probability(score_i, score_j):
    """Probability that entity i beats entity j."""
    return score_i / (score_i + score_j)

model.fit(data)
score_a = model.get_score("TeamA")
score_b = model.get_score("TeamB")

win_prob = pairwise_win_probability(score_a, score_b)
print(f"P(TeamA beats TeamB) = {win_prob:.2%}")
```

### 5. Uncertainty Quantification

```python
# Bootstrap for confidence intervals
from plackett_luce.utils import generate_synthetic_rankings
import numpy as np

def bootstrap_rankings(data, n_bootstrap=100):
    """Compute bootstrap confidence intervals."""
    bootstrap_scores = {entity: [] for entity in all_entities}
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [data[i] for i in np.random.choice(len(data), len(data))]
        
        model = PlackettLuceModel(model_type='full')
        model.fit(sample, verbose=False)
        
        for entity in all_entities:
            bootstrap_scores[entity].append(model.get_score(entity))
    
    # Compute percentiles
    results = {}
    for entity, scores in bootstrap_scores.items():
        results[entity] = {
            'mean': np.mean(scores),
            'ci_low': np.percentile(scores, 2.5),
            'ci_high': np.percentile(scores, 97.5),
        }
    
    return results

# Usage
ci_results = bootstrap_rankings(data, n_bootstrap=100)
for entity, stats in sorted(ci_results.items(), 
                           key=lambda x: x[1]['mean'], 
                           reverse=True)[:5]:
    print(f"{entity}: {stats['mean']:.3f} "
          f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]")
```

---

For more examples, see the [examples/](https://github.com/diogoribeiro7/efficient-plackett-luce/tree/main/examples) directory in the repository.
