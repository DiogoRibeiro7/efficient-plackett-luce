"""
Utility functions for data generation and analysis.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import kendalltau, spearmanr


def generate_synthetic_rankings(
    N: int = 100,
    M: int = 1000,
    K_min: int = 2,
    K_max: int = 10,
    seed: Optional[int] = None,
) -> List[Tuple]:
    """
    Generate synthetic rankings according to Plackett-Luce model.

    Parameters:
    -----------
    N : int
        Number of entities
    M : int
        Number of comparisons
    K_min, K_max : int
        Range of comparison sizes
    seed : int, optional
        Random seed

    Returns:
    --------
    List[Tuple] : Generated rankings
    """
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


def ranking_similarity(
    ranking1: List, ranking2: List, method: str = "kendall"
) -> float:
    """
    Compute similarity between two rankings.

    Parameters:
    -----------
    ranking1, ranking2 : List
        Rankings to compare
    method : str
        'kendall' for Kendall's tau or 'spearman' for Spearman's rho

    Returns:
    --------
    float : Similarity score
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
