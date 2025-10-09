# API Reference

Complete API documentation for efficient-plackett-luce.

## Core Classes

### PlackettLuceModel

Main class for Plackett-Luce ranking inference.

```python
class PlackettLuceModel(model_type='full', epsilon=1e-6, max_iterations=10000)
```

**Parameters:**

- `model_type` (str): Model variant to use
  - `'full'`: Full Plackett-Luce model (uses complete rankings)
  - `'position1'`: Position-1-breaking model (only uses winners)
- `epsilon` (float): Convergence threshold for iterative algorithm
- `max_iterations` (int): Maximum number of iterations before stopping

**Attributes:**

- `scores` (np.ndarray): Fitted scores for each entity
- `node_to_idx` (dict): Mapping from entity IDs to internal indices
- `is_fitted` (bool): Whether the model has been fitted

#### Methods

##### fit()

```python
fit(hyperedges, verbose=False) -> dict
```

Fit the model to comparison data.

**Parameters:**
- `hyperedges` (List[Tuple]): List of `(ranking, weight)` pairs
  - `ranking`: Tuple of entity IDs in decreasing order of performance
  - `weight`: Integer count of occurrences
- `verbose` (bool): Print convergence information

**Returns:**
- `dict`: Training statistics
  - `iterations`: Number of iterations to convergence
  - `time`: Training time in seconds
  - `final_convergence`: Final convergence metric value

**Example:**
```python
model = PlackettLuceModel()
stats = model.fit([
    (("TeamA", "TeamB", "TeamC"), 1),
    (("TeamB", "TeamC"), 2),
], verbose=True)
print(f"Converged in {stats['iterations']} iterations")
```

##### predict_probability()

```python
predict_probability(ranking) -> float
```

Compute probability of observing a specific ranking.

**Parameters:**
- `ranking` (Tuple): Ordered tuple of entity IDs

**Returns:**
- `float`: Probability between 0 and 1

**Example:**
```python
prob = model.predict_probability(("TeamA", "TeamB", "TeamC"))
print(f"P(A > B > C) = {prob:.4f}")
```

##### log_likelihood()

```python
log_likelihood(hyperedges) -> float
```

Compute log-likelihood of data under the fitted model.

**Parameters:**
- `hyperedges` (List[Tuple]): Comparison data

**Returns:**
- `float`: Log-likelihood value (always negative)

**Example:**
```python
train_ll = model.log_likelihood(train_data)
test_ll = model.log_likelihood(test_data)
```

##### get_ranking()

```python
get_ranking(top_k=None) -> List[Tuple[Any, float]]
```

Get ranking of entities by their scores.

**Parameters:**
- `top_k` (int, optional): Return only top k entities

**Returns:**
- `List[Tuple]`: List of `(entity_id, score)` pairs in decreasing score order

**Example:**
```python
top_10 = model.get_ranking(top_k=10)
for rank, (entity, score) in enumerate(top_10, 1):
    print(f"{rank}. {entity}: {score:.4f}")
```

##### get_score()

```python
get_score(node_id) -> float
```

Get score for a specific entity.

**Parameters:**
- `node_id`: Entity identifier

**Returns:**
- `float`: Score value

##### save()

```python
save(filepath)
```

Save fitted model to disk.

**Parameters:**
- `filepath` (str or Path): File path for saving

**Example:**
```python
model.save("my_model.pkl")
```

##### load()

```python
@classmethod
load(filepath) -> PlackettLuceModel
```

Load fitted model from disk.

**Parameters:**
- `filepath` (str or Path): Path to saved model

**Returns:**
- `PlackettLuceModel`: Loaded model instance

**Example:**
```python
model = PlackettLuceModel.load("my_model.pkl")
```

---

### ProjectedPlackettLuce

Projected pairwise version of Plackett-Luce model.

```python
class ProjectedPlackettLuce(model_type='full', epsilon=1e-6, max_iterations=10000)
```

Converts multi-body comparisons to pairwise comparisons before fitting. Useful for comparison with full PL model.

Inherits all methods from `PlackettLuceModel`.

**Example:**
```python
proj_model = ProjectedPlackettLuce(model_type='full')
proj_model.fit(data)
```

---

## Validation Functions

### cross_validate()

```python
cross_validate(hyperedges, model_type='full', n_splits=5, 
               compare_projected=True, random_state=None, verbose=False) -> dict
```

Perform k-fold cross-validation.

**Parameters:**
- `hyperedges` (List[Tuple]): Full dataset
- `model_type` (str): `'full'` or `'position1'`
- `n_splits` (int): Number of CV folds
- `compare_projected` (bool): Also evaluate projected model
- `random_state` (int, optional): Random seed
- `verbose` (bool): Print progress

**Returns:**
- `dict`: Cross-validation results
  - `pl_mean`: Mean log-likelihood for PL model
  - `pl_std`: Standard deviation
  - `pl_scores`: List of fold scores
  - `projected_mean`: Mean for projected model (if requested)
  - `projected_std`: Standard deviation for projected
  - `winner`: `'PL'` or `'Projected'`
  - `improvement`: Percentage improvement

**Example:**
```python
from plackett_luce import cross_validate

results = cross_validate(data, n_splits=5, verbose=True)
print(f"Winner: {results['winner']}")
print(f"Improvement: {results['improvement']:.2f}%")
```

### train_test_split()

```python
train_test_split(hyperedges, test_size=0.2, random_state=None, shuffle=True)
```

Split data into train and test sets.

**Parameters:**
- `hyperedges` (List[Tuple]): Dataset to split
- `test_size` (float): Proportion for test set (0.0 to 1.0)
- `random_state` (int, optional): Random seed
- `shuffle` (bool): Whether to shuffle before splitting

**Returns:**
- `Tuple[List, List]`: `(train_data, test_data)`

**Example:**
```python
from plackett_luce.validation import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=42)
```

### stratified_split_by_size()

```python
stratified_split_by_size(hyperedges, test_size=0.2, random_state=None)
```

Split while maintaining distribution of comparison sizes.

**Example:**
```python
from plackett_luce.validation import stratified_split_by_size

train, test = stratified_split_by_size(data, test_size=0.2)
```

### temporal_split()

```python
temporal_split(hyperedges, test_size=0.2)
```

Split temporally (first N% for train, last M% for test). Does not shuffle.

**Example:**
```python
from plackett_luce.validation import temporal_split

train, test = temporal_split(tournament_rounds, test_size=0.2)
```

---

## Utility Functions

### generate_synthetic_rankings()

```python
generate_synthetic_rankings(N=100, M=1000, K_min=2, K_max=10, seed=None)
```

Generate synthetic ranking data according to PL model.

**Parameters:**
- `N` (int): Number of entities
- `M` (int): Number of comparisons
- `K_min` (int): Minimum comparison size
- `K_max` (int): Maximum comparison size
- `seed` (int, optional): Random seed

**Returns:**
- `List[Tuple]`: Generated hyperedges

**Example:**
```python
from plackett_luce.utils import generate_synthetic_rankings

data = generate_synthetic_rankings(N=50, M=500, K_min=2, K_max=5, seed=42)
```

### ranking_similarity()

```python
ranking_similarity(ranking1, ranking2, method='kendall')
```

Compute similarity between two rankings.

**Parameters:**
- `ranking1`, `ranking2` (List): Rankings to compare
- `method` (str): `'kendall'` or `'spearman'`

**Returns:**
- `float`: Similarity score between -1 and 1

**Example:**
```python
from plackett_luce.utils import ranking_similarity

similarity = ranking_similarity(
    ["A", "B", "C", "D"],
    ["A", "C", "B", "D"],
    method='kendall'
)
```
