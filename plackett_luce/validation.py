"""
Model validation and cross-validation utilities.
"""

import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from sklearn.model_selection import KFold

from .core import PlackettLuceModel
from .projected import ProjectedPlackettLuce
from .utils import Hyperedge, normalize_hyperedges

_DEFAULT_SEED = 42


def _create_rng(
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]],
) -> np.random.Generator:
    """Create a NumPy Generator deterministic by default."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, 2**32 - 1))
        return np.random.default_rng(seed)
    if isinstance(random_state, (np.integer, int)):
        return np.random.default_rng(int(random_state))
    if random_state is None:
        return np.random.default_rng(_DEFAULT_SEED)
    raise TypeError(f"Unsupported random_state type: {type(random_state).__name__}")


def _rng_seed(
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]],
) -> int:
    """Derive an integer seed from assorted random_state inputs."""

    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**32 - 1))
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, 2**32 - 1))
    if isinstance(random_state, (np.integer, int)):
        return int(random_state)
    if random_state is None:
        return _DEFAULT_SEED
    raise TypeError(f"Unsupported random_state type: {type(random_state).__name__}")


def train_test_split(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    test_size: float = 0.2,
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
    shuffle: bool = True,
) -> Tuple[List[Hyperedge], List[Hyperedge]]:
    """Split ranking data into deterministic train and test subsets."""

    data = normalize_hyperedges(hyperedges)
    assert all(isinstance(edge, tuple) for edge in data)
    assert all(isinstance(edge[0], tuple) for edge in data)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}")

    n_samples = len(data)
    if n_samples == 1:
        # Degenerate case: choose split based on test_size proportion
        if test_size >= 0.5:
            return [], data.copy()
        return data.copy(), []

    if n_samples == 0:
        raise ValueError("hyperedges cannot be empty")

    desired_test = int(n_samples * test_size)
    n_test = desired_test
    if n_test <= 0:
        n_test = 1
    elif n_test >= n_samples:
        n_test = n_samples - 1

    n_train = n_samples - n_test

    rng = _create_rng(random_state)
    indices = np.arange(n_samples, dtype=int)
    if shuffle:
        indices = np.asarray(rng.permutation(indices), dtype=int)

    train_indices = indices[:n_train]
    test_indices = indices[n_train : n_train + n_test]

    train: List[Hyperedge] = [
        cast(Hyperedge, tuple(data[int(i)])) for i in np.sort(train_indices)
    ]
    test: List[Hyperedge] = [
        cast(Hyperedge, tuple(data[int(i)])) for i in np.sort(test_indices)
    ]

    assert all(isinstance(edge, tuple) and isinstance(edge[0], tuple) for edge in train)
    assert all(isinstance(edge, tuple) and isinstance(edge[0], tuple) for edge in test)
    assert len(train) + len(test) == n_samples
    train_index_set = {int(i) for i in train_indices}
    test_index_set = {int(i) for i in test_indices}
    assert train_index_set.isdisjoint(test_index_set)
    assert len(train_index_set | test_index_set) == n_samples
    if 0 < desired_test < n_samples:
        assert n_test == desired_test
    assert len(test) == n_test

    return train, test


def stratified_split_by_size(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    test_size: float = 0.2,
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
) -> Tuple[List[Hyperedge], List[Hyperedge]]:
    """Split hyperedges while maintaining distribution of comparison sizes."""

    data = normalize_hyperedges(hyperedges)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}")

    if not data:
        raise ValueError("hyperedges cannot be empty")

    rng = _create_rng(random_state)
    size_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, (ranking, _) in enumerate(data):
        size_groups[len(ranking)].append(idx)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for indices in size_groups.values():
        arr = np.array(indices, dtype=int)
        if arr.size == 0:
            continue
        if arr.size == 1:
            if test_size >= 0.5:
                test_indices.append(int(arr[0]))
            else:
                train_indices.append(int(arr[0]))
            continue

        shuffled = rng.permutation(arr)
        n_group_test = int(round(shuffled.size * test_size))
        if n_group_test <= 0:
            n_group_test = 1
        elif n_group_test >= shuffled.size:
            n_group_test = shuffled.size - 1
        test_part = shuffled[:n_group_test]
        train_part = shuffled[n_group_test:]
        test_indices.extend(int(i) for i in test_part)
        train_indices.extend(int(i) for i in train_part)

    # Ensure neither split is empty for datasets with more than one sample
    if not test_indices and len(data) > 1:
        moved = train_indices.pop()
        test_indices.append(moved)
    if not train_indices and len(data) > 1:
        moved = test_indices.pop()
        train_indices.append(moved)

    train = [data[i] for i in sorted(train_indices)]
    test = [data[i] for i in sorted(test_indices)]

    return train, test


def temporal_split(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    test_size: float = 0.2,
) -> Tuple[List[Hyperedge], List[Hyperedge]]:
    """Split hyperedges temporally (first portion for train, last for test)."""

    data = normalize_hyperedges(hyperedges)

    if not data:
        raise ValueError("hyperedges cannot be empty")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}")

    n_samples = len(data)
    if n_samples == 1:
        return ([], data.copy()) if test_size >= 0.5 else (data.copy(), [])

    desired_test = int(n_samples * test_size)
    n_test = desired_test
    if n_test <= 0:
        n_test = 1
    elif n_test >= n_samples:
        n_test = n_samples - 1

    n_train = n_samples - n_test

    train_data = data[:n_train]
    test_data = data[n_train : n_train + n_test]

    if 0 < desired_test < n_samples:
        assert n_test == desired_test

    return train_data, test_data


def cross_validate(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    model_type: str = "full",
    n_splits: int = 5,
    compare_projected: bool = True,
    random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, List[float], str, int]]:
    """Evaluate Plackett-Luce models via k-fold cross-validation."""

    if model_type not in PlackettLuceModel.MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {PlackettLuceModel.MODEL_TYPES}, got '{model_type}'"
        )

    data = normalize_hyperedges(hyperedges)
    dataset_size = len(data)
    assert all(isinstance(edge, tuple) and isinstance(edge[0], tuple) for edge in data)
    if dataset_size == 0:
        raise ValueError("hyperedges cannot be empty")

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if n_splits > dataset_size:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed dataset size ({dataset_size})")

    kf_seed = _rng_seed(random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=kf_seed)
    fold_rng = np.random.default_rng(kf_seed)

    scores_pl: List[float] = []
    scores_projected: Optional[List[float]] = [] if compare_projected else None

    fold_messages: List[str] = []
    indices = np.arange(dataset_size)

    tqdm_import: Optional[Any] = None
    iterator: Iterable[int]
    tqdm_bar: Optional[Any] = None
    if verbose:
        try:
            from tqdm import tqdm as tqdm_import  # type: ignore[import-not-found]
        except ImportError:
            print(f"Cross-validation over {n_splits} folds...")
            iterator = range(n_splits)
        else:
            tqdm_bar = tqdm_import(
                range(n_splits), desc="Cross-Validation", unit="fold", leave=True
            )
            iterator = cast(Iterable[int], tqdm_bar)
    else:
        iterator = range(n_splits)

    for fold_index, (train_idx, test_idx) in zip(iterator, kf.split(indices)):
        if tqdm_bar is not None:
            tqdm_bar.set_description(f"Cross-Validation: Fold {fold_index + 1}/{n_splits}")
        elif verbose:
            print(f"Fold {fold_index + 1}/{n_splits}...")

        train_idx_sorted = np.sort(train_idx)
        test_idx_sorted = np.sort(test_idx)

        train_data: List[Hyperedge] = [
            cast(Hyperedge, tuple(data[int(i)])) for i in train_idx_sorted
        ]
        test_data: List[Hyperedge] = [
            cast(Hyperedge, tuple(data[int(i)])) for i in test_idx_sorted
        ]

        assert all(isinstance(edge, tuple) and isinstance(edge[0], tuple) for edge in train_data)
        assert all(isinstance(edge, tuple) and isinstance(edge[0], tuple) for edge in test_data)

        fold_seed = int(fold_rng.integers(0, 2**32 - 1))

        model_pl = PlackettLuceModel(model_type=model_type, random_state=fold_seed)
        model_pl.fit(train_data, verbose=False)
        ll_pl = float(model_pl.log_likelihood(test_data))
        scores_pl.append(ll_pl)

        ll_proj: Optional[float] = None
        if compare_projected:
            assert scores_projected is not None
            model_proj = ProjectedPlackettLuce(model_type=model_type, random_state=fold_seed)
            model_proj.fit(train_data, verbose=False)
            ll_proj = float(model_proj.log_likelihood(test_data))
            scores_projected.append(ll_proj)

        if verbose:
            message = f"  Fold {fold_index + 1}/{n_splits} - PL: {ll_pl:.4f}"
            if compare_projected and ll_proj is not None:
                message += f", Projected: {ll_proj:.4f}"
            fold_messages.append(message)

    if tqdm_bar is not None:
        tqdm_bar.close()

    pl_mean = float(np.mean(scores_pl)) if scores_pl else float("nan")
    pl_std = float(np.std(scores_pl)) if scores_pl else float("nan")

    results: Dict[str, Union[float, List[float], str, int]] = {
        "pl_mean": pl_mean,
        "pl_std": pl_std,
        "pl_scores": scores_pl,
        "n_splits": n_splits,
        "model_type": model_type,
    }

    if compare_projected:
        assert scores_projected is not None
        projected_mean = float(np.mean(scores_projected)) if scores_projected else float("nan")
        projected_std = float(np.std(scores_projected)) if scores_projected else float("nan")
        results["projected_mean"] = projected_mean
        results["projected_std"] = projected_std
        results["projected_scores"] = scores_projected

        if pl_mean > projected_mean:
            denominator = abs(projected_mean) if projected_mean != 0 else 1.0
            results["winner"] = "PL"
            results["improvement"] = (pl_mean - projected_mean) / denominator * 100
        else:
            denominator = abs(pl_mean) if pl_mean != 0 else 1.0
            results["winner"] = "Projected"
            results["improvement"] = (projected_mean - pl_mean) / denominator * 100

    if verbose:
        for message in fold_messages:
            print(message)
        print(f"Mean PL: {pl_mean:.4f} +/- {pl_std:.4f}")
        if compare_projected and "projected_mean" in results:
            projected_mean = cast(float, results["projected_mean"])
            projected_std = cast(float, results["projected_std"])
            improvement = cast(float, results.get("improvement", 0.0))
            print(f"Mean Projected: {projected_mean:.4f} +/- {projected_std:.4f}")
            print(f"Winner: {results.get('winner', 'PL')} ({improvement:.1f}% improvement)")

    return cast(Dict[str, Union[float, List[float], str, int]], results)


def leave_one_out_cv(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    model_type: str = "full",
    verbose: bool = False,
) -> Dict[str, Union[float, List[float], int, str]]:
    """Perform leave-one-out cross-validation on ranking data."""

    data = normalize_hyperedges(hyperedges)
    n = len(data)
    if n == 0:
        raise ValueError("hyperedges cannot be empty")

    if n == 1:
        warnings.warn(
            "Leave-one-out CV with single sample will train on empty dataset",
            RuntimeWarning,
        )

    scores: List[float] = []

    iterator: Iterable[int]
    tqdm_import: Optional[Any] = None
    progress_bar: Optional[Any] = None
    if verbose:
        try:
            from tqdm import tqdm as tqdm_import  # type: ignore[import-not-found]
        except ImportError:
            print(f"Processing {n} folds...")
            iterator = range(n)
        else:
            progress_bar = tqdm_import(range(n), desc="Leave-One-Out CV", unit="fold", leave=True)
            iterator = cast(Iterable[int], progress_bar)
    else:
        iterator = range(n)

    for i in iterator:
        if progress_bar is not None:
            progress_bar.set_postfix({"Fold": f"{i + 1}/{n}"})
        elif verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n):
            print(f"Processing {i + 1}/{n}...")

        train_data = data[:i] + data[i + 1 :]
        test_data = [data[i]]

        if train_data:
            model = PlackettLuceModel(model_type=model_type, random_state=_DEFAULT_SEED + i)
            model.fit(train_data, verbose=False)
            ll = float(model.log_likelihood(test_data))
        else:
            ll = 0.0
        scores.append(ll)

    if progress_bar is not None:
        progress_bar.close()

    return cast(
        Dict[str, Union[float, List[float], int, str]],
        {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores,
            "n_samples": n,
            "model_type": model_type,
        },
    )
