"""
Model validation and cross-validation utilities.
"""

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold

from .core import PlackettLuceModel
from .projected import ProjectedPlackettLuce


def train_test_split(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Split ranking data into train and test subsets.

    Provides a lightweight alternative to
    :func:`sklearn.model_selection.train_test_split` tailored to hyperedge
    (ranking, weight) pairs. The function preserves ranking tuples without the
    additional metadata that scikit-learn introduces.

    Parameters
    ----------
    hyperedges : List[Tuple[Tuple, Union[int, float]]]
        Sequence of (ranking, weight) pairs.
    test_size : float, default=0.2
        Fraction of data assigned to the test split; must satisfy
        ``0.0 < test_size < 1.0``.
    random_state : int, optional
        Seed for the NumPy random generator used when shuffling.
    shuffle : bool, default=True
        Whether to shuffle data before splitting. Recommended for IID ranking
        datasets; set to ``False`` for temporally ordered competitions and use
        :func:`temporal_split` instead.

    Returns
    -------
    Tuple[List[Tuple], List[Tuple]]
        Train and test subsets as lists of hyperedges.

    Raises
    ------
    ValueError
        If ``test_size`` is outside ``(0.0, 1.0)`` or ``hyperedges`` is empty.

    Examples
    --------
    >>> from plackett_luce.validation import train_test_split
    >>> data = [(("A", "B", "C"), 1), (("B", "C", "A"), 1), (("C", "A", "B"), 1)]
    >>> train, test = train_test_split(data, test_size=0.33, random_state=0)
    >>> len(train), len(test)
    (2, 1)

    Notes
    -----
    * Shuffling mitigates bias from grouped or chronological ordering. Disable
      only when domain constraints require deterministic ordering.
    * The split is deterministic when ``random_state`` is provided.

    See Also
    --------
    temporal_split : Deterministic chronological split for time-ordered data.
    stratified_split_by_size : Stratified splitting on comparison size.
    cross_validate : Full k-fold cross-validation for model assessment.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}")

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
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
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
    if not hyperedges:
        raise ValueError("hyperedges cannot be empty")

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}")

    if random_state is not None:
        if not isinstance(random_state, int):
            raise TypeError(f"random_state must be an integer or None, got {random_state!r}")
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
    hyperedges: List[Tuple[Tuple, Union[int, float]]], test_size: float = 0.2
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
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    model_type: str = "full",
    n_splits: int = 5,
    compare_projected: bool = True,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, List[float], str, int]]:
    """
    Evaluate Plackett-Luce models via k-fold cross-validation.

    The dataset is partitioned into ``n_splits`` disjoint folds. Each iteration
    trains the model on ``n_splits - 1`` folds and evaluates log-likelihood on
    the held-out fold. Enabling ``compare_projected=True`` additionally fits the
    Projected Plackett-Luce model per fold, highlighting the benefit of modeling
    full rankings rather than only pairwise outcomes.

    Parameters
    ----------
    hyperedges : List[Tuple[Tuple, Union[int, float]]]
        Full dataset of (ranking, weight) pairs.
    model_type : str, default='full'
        Model variant to evaluate, typically ``'full'`` or ``'position1'``.
    n_splits : int, default=5
        Number of cross-validation folds; must satisfy
        ``2 <= n_splits <= len(hyperedges)``.
    compare_projected : bool, default=True
        If ``True``, evaluate the Projected Plackett-Luce model alongside the full
        model to quantify differences between higher-order and pairwise modeling.
    random_state : int, optional
        Seed forwarded to :class:`sklearn.model_selection.KFold` when shuffling.
    verbose : bool, default=False
        Print progress information. If :mod:`tqdm` is installed, fold progress is
        displayed via a progress bar.

    Returns
    -------
    Dict[str, Union[float, List[float], str, int]]
        Summary statistics including:

        ``'pl_mean'`` : Mean PL log-likelihood across folds.
        ``'pl_std'`` : Standard deviation of PL log-likelihoods.
        ``'pl_scores'`` : List of per-fold PL log-likelihoods.
        ``'projected_mean'`` : Mean projected log-likelihood (if computed).
        ``'projected_std'`` : Standard deviation of projected log-likelihoods.
        ``'projected_scores'`` : List of per-fold projected log-likelihoods.
        ``'n_splits'`` : Number of folds used.
        ``'model_type'`` : Model type evaluated.
        ``'winner'`` : Model with the higher mean log-likelihood when both are evaluated.
        ``'improvement'`` : Percentage improvement of the winner over the alternative.

    Raises
    ------
    ValueError
        If ``model_type`` is unsupported, ``n_splits`` is less than 2, or
        ``n_splits`` exceeds ``len(hyperedges)``.

    Warnings
    --------
    Small datasets can yield high-variance fold estimates, potentially
    exaggerating differences between models. Consider leave-one-out
    cross-validation or collecting more comparisons when ``len(hyperedges)``
    is small.

    Examples
    --------
    >>> from plackett_luce.utils import generate_synthetic_rankings
    >>> from plackett_luce.validation import cross_validate
    >>> rankings = generate_synthetic_rankings(N=20, M=80, K_min=2, K_max=5, seed=0)
    >>> results = cross_validate(rankings, n_splits=4, compare_projected=True)
    >>> results['winner'], round(results['improvement'], 2)  # doctest: +SKIP
    ('PL', 2.35)

    Notes
    -----
    * K-fold CV balances bias and variance but requires ``O(n_splits)`` model
      fits (``2 * n_splits`` when ``compare_projected=True``).
    * The ``winner`` field names the model with the larger mean log-likelihood;
      ``improvement`` reports its relative advantage in percent (negative values
      indicate the Projected model performed better).
    * Set ``compare_projected=False`` when runtime is a concern or only the full
      model matters.

    See Also
    --------
    leave_one_out_cv : Exhaustive leave-one-out evaluation.
    train_test_split : Simple train/test split for quick validation.
    PlackettLuceModel : Implementation of the Plackett-Luce family.

    """
    if model_type not in PlackettLuceModel.MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {PlackettLuceModel.MODEL_TYPES}, got '{model_type}'"
        )

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    dataset_size = len(hyperedges)
    if n_splits > dataset_size:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed dataset size ({dataset_size})")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores_pl: List[float] = []
    scores_projected: Optional[List[float]] = [] if compare_projected else None

    hyperedges_array = np.array(hyperedges, dtype=object)

    progress_iterator: Union[range, "tqdm"]  # type: ignore[name-defined]
    tqdm_cls = None
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
        except ImportError:
            print(f"Cross-validation over {n_splits} folds...")
            progress_iterator = range(n_splits)
        else:
            progress_iterator = tqdm(
                range(n_splits),
                desc="Cross-Validation",
                unit="fold",
                leave=True,
            )
            tqdm_cls = tqdm
    else:
        progress_iterator = range(n_splits)

    split_generator = kf.split(hyperedges_array)
    fold_messages: List[str] = []

    for fold_index in progress_iterator:
        train_idx, test_idx = next(split_generator)

        if tqdm_cls is not None:
            progress_iterator.set_description(  # type: ignore[attr-defined]
                f"Cross-Validation: Fold {fold_index + 1}/{n_splits}"
            )
        elif verbose:
            print(f"Fold {fold_index + 1}/{n_splits}...")

        train_data = hyperedges_array[train_idx].tolist()
        test_data = hyperedges_array[test_idx].tolist()

        # Optional nested progress for model fitting
        fold_bar = None
        if verbose and tqdm_cls is not None:
            total_models = 2 if compare_projected else 1
            fold_bar = tqdm_cls(
                total=total_models,
                desc=f"Fold {fold_index + 1}/{n_splits}",
                unit="model",
                leave=False,
            )

        # Fit full PL model
        model_pl = PlackettLuceModel(model_type=model_type)
        model_pl.fit(train_data, verbose=False)
        ll_pl = model_pl.log_likelihood(test_data)
        scores_pl.append(ll_pl)

        if fold_bar is not None:
            fold_bar.set_postfix({"Model": "PL"})
            fold_bar.update(1)

        # Fit projected model if requested
        ll_proj: Optional[float] = None
        if compare_projected:
            assert scores_projected is not None
            model_proj = ProjectedPlackettLuce(model_type=model_type)
            model_proj.fit(train_data, verbose=False)
            ll_proj = model_proj.log_likelihood(test_data)
            scores_projected.append(ll_proj)

            if fold_bar is not None:
                fold_bar.set_postfix({"Model": "Projected"})
                fold_bar.update(1)

        if fold_bar is not None:
            fold_bar.close()

        if verbose:
            if compare_projected and ll_proj is not None:
                fold_messages.append(
                    f"  Fold {fold_index + 1}/{n_splits} - PL: {ll_pl:.4f}, "
                    f"Projected: {ll_proj:.4f}"
                )
            else:
                fold_messages.append(f"  Fold {fold_index + 1}/{n_splits} - PL: {ll_pl:.4f}")

    results = {
        "pl_mean": np.mean(scores_pl),
        "pl_std": np.std(scores_pl),
        "pl_scores": scores_pl,
        "n_splits": n_splits,
        "model_type": model_type,
    }

    if compare_projected:
        assert scores_projected is not None
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
                (results["projected_mean"] - results["pl_mean"]) / abs(results["pl_mean"]) * 100
            )

        results["improvement"] = improvement

    if verbose:
        if tqdm_cls is not None and hasattr(progress_iterator, "close"):
            progress_iterator.close()  # type: ignore[call-arg]
        for message in fold_messages:
            print(message)
        print(f"Mean PL: {results['pl_mean']:.4f} +/- {results['pl_std']:.4f}")
        if compare_projected:
            print(
                f"Mean Projected: {results['projected_mean']:.4f} +/- "
                f"{results['projected_std']:.4f}"
            )
            print(f"Winner: {results['winner']} " f"({results['improvement']:.1f}% improvement)")

    return results


def leave_one_out_cv(
    hyperedges: List[Tuple[Tuple, Union[int, float]]],
    model_type: str = "full",
    verbose: bool = False,
) -> Dict[str, Union[float, List[float], int, str]]:
    """
    Perform leave-one-out cross-validation on ranking data.

    Leave-one-out cross-validation (LOO-CV) iteratively holds out each comparison
    once as the test set while fitting the model on the remaining ``M - 1``
    comparisons. Compared with k-fold CV it yields nearly unbiased estimates but
    incurs higher variance and computational cost proportional to the number of
    comparisons.

    Parameters
    ----------
    hyperedges : List[Tuple[Tuple, Union[int, float]]]
        Complete dataset of (ranking, weight) pairs.
    model_type : str, default='full'
        Plackett-Luce model variant to train.
    verbose : bool, default=False
        When ``True``, display progress (using :mod:`tqdm` if available).

    Returns
    -------
    Dict[str, Union[float, List[float], int, str]]
        Summary metrics analogous to :func:`cross_validate`:

        ``'mean'`` : Mean held-out log-likelihood.
        ``'std'`` : Standard deviation across held-out comparisons.
        ``'scores'`` : List of per-comparison log-likelihoods.
        ``'n_samples'`` : Number of evaluated comparisons.
        ``'model_type'`` : Evaluated model variant.

    Raises
    ------
    ValueError
        If ``hyperedges`` is empty.

    Warnings
    --------
    Issues a warning when ``len(hyperedges) == 1`` because the training set
    becomes empty. Runtime is ``O(M)`` model fits, which can be substantial for
    large datasets.

    Examples
    --------
    >>> from plackett_luce.utils import generate_synthetic_rankings
    >>> from plackett_luce.validation import leave_one_out_cv
    >>> rankings = generate_synthetic_rankings(N=8, M=10, K_min=2, K_max=3, seed=1)
    >>> results = leave_one_out_cv(rankings, model_type="full")
    >>> results["n_samples"] == len(rankings)
    True

    Notes
    -----
    * Prefer LOO-CV on very small datasets where k-fold splits would leave too
      little training data per fold.
    * For moderate or large datasets, k-fold CV offers a better trade-off between
      computational cost and variance.

    See Also
    --------
    cross_validate : K-fold cross-validation with optional model comparison.
    train_test_split : Simple train/test split for quick evaluation.
    """
    if not hyperedges:
        raise ValueError("hyperedges cannot be empty")

    if len(hyperedges) == 1:
        warnings.warn(
            "Leave-one-out CV with single sample will train on empty dataset",
            RuntimeWarning,
        )

    n = len(hyperedges)
    scores = []

    iterator: Union[range, "tqdm"]  # type: ignore[name-defined]
    progress_bar = None
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
        except ImportError:
            print(f"Processing {n} folds...")
            iterator = range(n)
        else:
            progress_bar = tqdm(
                range(n),
                desc="Leave-One-Out CV",
                unit="fold",
                leave=True,
            )
            iterator = progress_bar
    else:
        iterator = range(n)

    for i in iterator:
        if progress_bar is not None:
            progress_bar.set_postfix({"Fold": f"{i + 1}/{n}"})
        elif verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n):
            print(f"Processing {i + 1}/{n}...")

        # Leave one out
        train_data = hyperedges[:i] + hyperedges[i + 1 :]
        test_data = [hyperedges[i]]

        # Fit and evaluate
        model = PlackettLuceModel(model_type=model_type)
        model.fit(train_data, verbose=False)
        ll = model.log_likelihood(test_data)
        scores.append(ll)

    if progress_bar is not None:
        progress_bar.close()

    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "scores": scores,
        "n_samples": n,
        "model_type": model_type,
    }
