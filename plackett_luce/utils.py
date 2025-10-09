"""Utility functions for data generation and analysis."""

from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
from scipy.stats import kendalltau, spearmanr

Hyperedge = Tuple[Tuple[Any, ...], Union[int, float]]


def _python_scalar(value: Union[int, float, np.generic]) -> Union[int, float]:
    """Convert NumPy scalar values to Python scalars while preserving type."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_ranking_tuple(ranking: Any) -> Tuple[Any, ...]:
    """Convert assorted ranking structures into a tuple-only representation."""

    current = ranking
    # Unwrap redundant single-item containers, e.g. [[[1, 2, 3]]]
    while isinstance(current, (list, tuple)) and len(current) == 1 and isinstance(
        current[0], (list, tuple)
    ):
        current = current[0]

    if isinstance(current, (list, tuple)):
        coerced = tuple(
            _coerce_ranking_tuple(item) if isinstance(item, (list, tuple)) else item
            for item in current
        )
    else:
        coerced = (current,)

    assert isinstance(coerced, tuple)
    return coerced


def normalize_hyperedges(hyperedges: Iterable[Any]) -> List[Hyperedge]:
    """Normalize hyperedge structures to canonical ``(ranking, weight)`` pairs."""

    normalized: List[Hyperedge] = []

    for idx, item in enumerate(hyperedges):
        current = item

        # Unwrap single-item nesting such as [[(..., weight)]]
        while (
            isinstance(current, list)
            and len(current) == 1
            and isinstance(current[0], (list, tuple))
        ):
            current = current[0]

        if not isinstance(current, (list, tuple)) or len(current) != 2:
            raise ValueError(
                f"Hyperedge at index {idx} must be a (ranking, weight) pair, got {current!r}"
            )

        ranking_raw, weight_raw = current

        ranking_tuple = _coerce_ranking_tuple(ranking_raw)
        assert isinstance(ranking_tuple, tuple)

        if len(ranking_tuple) < 2:
            raise ValueError(f"Ranking must contain at least two entities at index {idx}")

        seen = set()
        unique_ranking: List[Any] = []
        for node in ranking_tuple:
            if node in seen:
                raise ValueError(f"Ranking contains duplicate node {node!r} at index {idx}")
            seen.add(node)
            unique_ranking.append(node)

        ranking = tuple(unique_ranking)

        weight = _python_scalar(weight_raw)
        if not isinstance(weight, (int, float)):
            raise ValueError(f"Weight must be numeric at index {idx}, got {type(weight).__name__}")
        if weight <= 0:
            raise ValueError(f"Weight must be positive at index {idx}, got {weight}")

        normalized.append((ranking, weight))

    if not normalized:
        raise ValueError("Hyperedges cannot be empty")

    return normalized


def generate_synthetic_rankings(
    N: int = 100,
    M: int = 1000,
    K_min: int = 2,
    K_max: int = 10,
    seed: Union[int, None] = None,
) -> List[Tuple[Tuple[int, ...], float]]:
    """Generate synthetic rankings according to the Plackett-Luce model."""

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
    if K_min > K_max:
        raise ValueError(f"K_min ({K_min}) cannot exceed K_max ({K_max})")
    if K_max > N:
        warnings.warn(
            f"K_max ({K_max}) cannot exceed N ({N}); clipping to N.",
            RuntimeWarning,
        )
        K_max = N

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

    rng = np.random.default_rng(seed if seed is not None else 42)

    # Generate true scores from logistic distribution
    u = rng.uniform(0, 1, N)
    true_scores = u / (1 - u)
    true_scores /= np.exp(np.mean(np.log(true_scores)))

    hyperedges: List[Tuple[Tuple[int, ...], float]] = []
    for _ in range(M):
        K = int(rng.integers(K_min, K_max + 1))
        nodes = rng.choice(N, size=K, replace=False)

        node_scores = [(node, true_scores[node]) for node in nodes]
        node_scores.sort(key=lambda x: x[1], reverse=True)
        ranking = tuple(node for node, _ in node_scores)

        hyperedge = (ranking, float(1))
        assert isinstance(hyperedge[0], tuple)
        assert isinstance(hyperedge[1], float)
        hyperedges.append(hyperedge)

    return hyperedges


def ranking_similarity(ranking1: List[Any], ranking2: List[Any], method: str = "kendall") -> float:
    """Compute similarity between two rankings."""

    pos1 = {item: i for i, item in enumerate(ranking1)}
    pos2 = {item: i for i, item in enumerate(ranking2)}

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

    if corr is None:
        return 0.0

    corr_value = float(corr)
    if np.isnan(corr_value):
        return 0.0
    return corr_value
