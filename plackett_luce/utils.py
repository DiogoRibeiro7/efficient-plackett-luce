"""
Utility functions for data generation and analysis.
"""

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr


def generate_synthetic_rankings(
    N: int = 100,
    M: int = 1000,
    K_min: int = 2,
    K_max: int = 10,
    seed: Optional[int] = None,
) -> List[Tuple[Tuple[int, ...], int]]:
    """
    Generate synthetic rankings according to Plackett-Luce model.

    Creates synthetic tournament/comparison data by:

    1. Generating true underlying scores from a logistic distribution
    2. Sampling K entities uniformly for each comparison
    3. Ordering them by their scores to create rankings

    Parameters
    ----------
    N : int
        Number of entities (items/players/teams). Must be >= K_max.
    M : int
        Number of comparisons to generate. Must be >= 1.
    K_min : int
        Minimum comparison size (number of entities per comparison).
        Must be >= 2.
    K_max : int
        Maximum comparison size. Must be >= K_min and <= N.
    seed : int, optional
        Random seed for reproducibility. If None, results are non-deterministic.

    Returns
    -------
    List[Tuple[Tuple[int, ...], int]]
        List of (ranking, weight) tuples where:

        * ranking: Tuple of entity indices in decreasing order of performance
        * weight: Always 1 in current implementation

    Raises
    ------
    ValueError
        If N < 1, M < 1, K_min < 2, K_max > N, K_min > K_max,
        or any parameter fails validation.

    Warnings
    --------
    Warns if M < 10 (dataset may be too small for reliable inference) or if
    K_max is close to N (comparisons approach full population size).

    Examples
    --------
    >>> from plackett_luce.utils import generate_synthetic_rankings
    >>>
    >>> # Generate simple pairwise comparisons
    >>> data = generate_synthetic_rankings(N=5, M=10, K_min=2, K_max=2, seed=42)
    >>> len(data)
    10
    >>> data[0]
    ((3, 1), 1)  # Entity 3 beats entity 1
    >>>
    >>> # Generate multi-body comparisons
    >>> data = generate_synthetic_rankings(N=20, M=100, K_min=3, K_max=7, seed=42)
    >>> all(3 <= len(ranking) <= 7 for ranking, _ in data)
    True

    Notes
    -----
    * All entity IDs are integers from 0 to N-1.
    * Rankings never contain duplicate entities.
    * Scores are normalized such that their geometric mean equals 1.
    * The generation process follows the true Plackett-Luce model.

    See Also
    --------
    plackett_luce.validation.cross_validate
        Perform k-fold cross-validation on ranking data.
    plackett_luce.core.PlackettLuceModel.fit
        Fit a model to ranking data.
    """
    for name, value in {"N": N, "M": M, "K_min": K_min, "K_max": K_max}.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer, got {value}")
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer, got {value}")

    if N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    if M < 1:
        raise ValueError(f"M must be a positive integer, got {M}")
    if K_min < 2:
        raise ValueError(f"K_min must be at least 2 for meaningful comparisons, got {K_min}")
    if K_max > N:
        raise ValueError(f"K_max ({K_max}) cannot exceed N ({N})")
    if K_min > K_max:
        raise ValueError(f"K_min ({K_min}) cannot exceed K_max ({K_max})")

    if M < 10:
        warnings.warn(
            f"Very small dataset: M ({M}) < 10; rankings may be unreliable.",
            RuntimeWarning,
        )
    if K_max / N >= 0.9:
        warnings.warn(
            f"K_max ({K_max}) is close to N ({N}); comparisons may include most entities.",
            RuntimeWarning,
        )

    if seed is not None:
        np.random.seed(seed)

    # Generate true scores from logistic distribution
    u = np.random.uniform(0, 1, N)
    true_scores = u / (1 - u)
    true_scores /= np.exp(np.mean(np.log(true_scores)))

    hyperedges = []
    for _ in range(M):
        K = np.random.randint(K_min, K_max + 1)
        nodes = np.random.choice(N, K, replace=False)

        # Order by scores
        node_scores = [(node, true_scores[node]) for node in nodes]
        node_scores.sort(key=lambda x: x[1], reverse=True)
        ranking = tuple(node for node, _ in node_scores)

        hyperedges.append((ranking, 1))

    return hyperedges


def ranking_similarity(ranking1: List[Any], ranking2: List[Any], method: str = "kendall") -> float:
    """
    Compute similarity between two rankings.

    Parameters
    ----------
    ranking1 : List[Any]
        First ranking as an ordered list of items. Items must be hashable.
    ranking2 : List[Any]
        Second ranking as an ordered list of items.
    method : str, default="kendall"
        Similarity metric to use. Supported values are ``"kendall"`` for
        Kendall's tau and ``"spearman"`` for Spearman's rho.

    Returns
    -------
    float
        Similarity score between the rankings. Returns 0.0 if the rankings
        share no common items. ``numpy.nan`` may be returned by the underlying
        statistic when the correlation is undefined.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.

    Examples
    --------
    >>> from plackett_luce.utils import ranking_similarity
    >>> ranking_similarity(["A", "B", "C"], ["A", "B", "C"], method="kendall")
    1.0
    >>> ranking_similarity(["A", "B", "C"], ["C", "B", "A"], method="spearman")
    -1.0
    >>> ranking_similarity(["A", "B"], ["C", "D"])
    0.0

    Notes
    -----
    The similarity is computed over the intersection of items in both rankings.
    Items appearing in only one ranking are ignored.
    """
    # Convert rankings to positions
    pos1 = {item: i for i, item in enumerate(ranking1)}
    pos2 = {item: i for i, item in enumerate(ranking2)}

    # Get common items
    common = set(ranking1) & set(ranking2)
    if not common:
        return 0.0

    ranks1 = [pos1[item] for item in common]
    ranks2 = [pos2[item] for item in common]

    if method == "kendall":
        corr, _ = kendalltau(ranks1, ranks2)
    elif method == "spearman":
        corr, _ = spearmanr(ranks1, ranks2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr
