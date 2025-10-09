"""
Model validation and cross-validation utilities.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import KFold
from collections import defaultdict

from .core import PlackettLuceModel
from .projected import ProjectedPlackettLuce


def train_test_split(
    hyperedges: List[Tuple],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Split hyperedges into train and test sets.

    Parameters:
    -----------
    hyperedges : List[Tuple]
        List of (ranking, weight) pairs
    test_size : float
        Proportion of dataset to include in test split (0.0 to 1.0)
    random_state : int, optional
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle data before splitting

    Returns:
    --------
    Tuple[List[Tuple], List[Tuple]] : (train_data, test_data)

    Examples:
    ---------
    >>> data = [(("A", "B", "C"), 1), (("B", "C", "A"), 1)]
    >>> train, test = train_test_split(data, test_size=0.3, random_state=42)
    >>> len(test) / len(data)
    0.3
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    if not hyperedges:
        raise ValueError("hyperedges cannot be empty")

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Convert to array for easier manipulation
    hyperedges_array = np.array(hyperedges, dtype=object)
    n_samples = len(hyperedges_array)

    # Create indices
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Calculate split point
    n_test = int(np.ceil(n_samples * test_size))
    n_train = n_samples - n_test

    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
    train_data = hyperedges_array[train_indices].tolist()
    test_data = hyperedges_array[test_indices].tolist()

    return train_data, test_data


def stratified_split_by_size(
    hyperedges: List[Tuple], test_size: float = 0.2, random_state: Optional[int] = None
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Split hyperedges while maintaining distribution of comparison sizes.

    Useful when you want train/test sets to have similar distributions
    of K (number of entities in each comparison).

    Parameters:
    -----------
    hyperedges : List[Tuple]
        List of (ranking, weight) pairs
    test_size : float
        Proportion for test set
    random_state : int, optional
        Random seed

    Returns:
    --------
    Tuple[List[Tuple], List[Tuple]] : (train_data, test_data)

    Examples:
    ---------
    >>> data = [(("A", "B"), 1), (("C", "D"), 1),
    ...         (("E", "F", "G"), 1), (("H", "I", "J"), 1)]
    >>> train, test = stratified_split_by_size(data, test_size=0.5, random_state=42)
    # Test set will have similar proportion of size-2 and size-3 comparisons
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Group by comparison size
    size_groups = defaultdict(list)
    for i, (ranking, weight) in enumerate(hyperedges):
        K = len(ranking)
        size_groups[K].append(i)

    train_indices = []
    test_indices = []

    # Split each group
    for K, indices in size_groups.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        n_test = int(np.ceil(len(indices) * test_size))
        n_train = len(indices) - n_test

        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])

    # Convert to data
    hyperedges_array = np.array(hyperedges, dtype=object)
    train_data = hyperedges_array[train_indices].tolist()
    test_data = hyperedges_array[test_indices].tolist()

    return train_data, test_data


def temporal_split(
    hyperedges: List[Tuple], test_size: float = 0.2
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Split hyperedges temporally (first N% for train, last M% for test).

    Useful for time-series data where you want to predict future rankings
    from past rankings. Does NOT shuffle.

    Parameters:
    -----------
    hyperedges : List[Tuple]
        List of (ranking, weight) pairs in temporal order
    test_size : float
        Proportion for test set

    Returns:
    --------
    Tuple[List[Tuple], List[Tuple]] : (train_data, test_data)

    Examples:
    ---------
    >>> # Tournament rounds in chronological order
    >>> rounds = [round1, round2, round3, round4, round5]
    >>> train, test = temporal_split(rounds, test_size=0.2)
    # Train on first 80%, test on last 20%
    """
    n_samples = len(hyperedges)
    n_train = int(n_samples * (1 - test_size))

    train_data = hyperedges[:n_train]
    test_data = hyperedges[n_train:]

    return train_data, test_data


def cross_validate(
    hyperedges: List[Tuple],
    model_type: str = "full",
    n_splits: int = 5,
    compare_projected: bool = True,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """
    Perform k-fold cross-validation.

    Parameters:
    -----------
    hyperedges : List[Tuple]
        Full dataset of (ranking, weight) pairs
    model_type : str
        'full' or 'position1'
    n_splits : int
        Number of CV folds
    compare_projected : bool
        Also evaluate projected pairwise model
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress

    Returns:
    --------
    dict : Cross-validation results with keys:
        - 'pl_mean': Mean log-likelihood for PL model
        - 'pl_std': Std dev of log-likelihood
        - 'pl_scores': List of scores for each fold
        - 'projected_mean': Mean for projected model (if compare_projected=True)
        - 'projected_std': Std dev for projected model
        - 'projected_scores': List of scores for projected model
        - 'winner': 'PL' or 'Projected' based on mean score
        - 'improvement': Percentage improvement of winner

    Examples:
    ---------
    >>> results = cross_validate(data, n_splits=5, verbose=True)
    >>> print(f"PL model: {results['pl_mean']:.4f} ± {results['pl_std']:.4f}")
    >>> print(f"Winner: {results['winner']}")
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores_pl = []
    scores_projected = [] if compare_projected else None

    hyperedges_array = np.array(hyperedges, dtype=object)

    for fold, (train_idx, test_idx) in enumerate(kf.split(hyperedges_array)):
        if verbose:
            print(f"Fold {fold + 1}/{n_splits}...")

        train_data = hyperedges_array[train_idx].tolist()
        test_data = hyperedges_array[test_idx].tolist()

        # Fit full PL model
        model_pl = PlackettLuceModel(model_type=model_type)
        model_pl.fit(train_data, verbose=False)
        ll_pl = model_pl.log_likelihood(test_data)
        scores_pl.append(ll_pl)

        if verbose:
            print(f"  PL log-likelihood: {ll_pl:.4f}")

        # Fit projected model if requested
        if compare_projected:
            model_proj = ProjectedPlackettLuce(model_type=model_type)
            model_proj.fit(train_data, verbose=False)
            ll_proj = model_proj.log_likelihood(test_data)
            scores_projected.append(ll_proj)

            if verbose:
                print(f"  Projected log-likelihood: {ll_proj:.4f}")
                print(f"  Difference: {ll_pl - ll_proj:.4f}")

    results = {
        "pl_mean": np.mean(scores_pl),
        "pl_std": np.std(scores_pl),
        "pl_scores": scores_pl,
        "n_splits": n_splits,
        "model_type": model_type,
    }

    if compare_projected:
        results.update(
            {
                "projected_mean": np.mean(scores_projected),
                "projected_std": np.std(scores_projected),
                "projected_scores": scores_projected,
            }
        )

        # Determine winner
        if results["pl_mean"] > results["projected_mean"]:
            results["winner"] = "PL"
            improvement = (
                (results["pl_mean"] - results["projected_mean"])
                / abs(results["projected_mean"])
                * 100
            )
        else:
            results["winner"] = "Projected"
            improvement = (
                (results["projected_mean"] - results["pl_mean"])
                / abs(results["pl_mean"])
                * 100
            )

        results["improvement"] = improvement

    return results


def leave_one_out_cv(
    hyperedges: List[Tuple], model_type: str = "full", verbose: bool = False
) -> Dict:
    """
    Perform leave-one-out cross-validation.

    Each comparison is held out once as the test set.
    Useful for small datasets but computationally expensive.

    Parameters:
    -----------
    hyperedges : List[Tuple]
        Full dataset
    model_type : str
        'full' or 'position1'
    verbose : bool
        Print progress

    Returns:
    --------
    dict : Results similar to cross_validate()
    """
    n = len(hyperedges)
    scores = []

    for i in range(n):
        if verbose and i % 10 == 0:
            print(f"Processing {i + 1}/{n}...")

        # Leave one out
        train_data = hyperedges[:i] + hyperedges[i + 1 :]
        test_data = [hyperedges[i]]

        # Fit and evaluate
        model = PlackettLuceModel(model_type=model_type)
        model.fit(train_data, verbose=False)
        ll = model.log_likelihood(test_data)
        scores.append(ll)

    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "scores": scores,
        "n_samples": n,
        "model_type": model_type,
    }
