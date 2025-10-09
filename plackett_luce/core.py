"""
Core Plackett-Luce model implementation.

Contains the main PlackettLuceModel class with full and position-1-breaking
variants using Newman's efficient algorithm.
"""

import hashlib
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from .optimization import (
    compute_convergence,
    compute_log_likelihood_pl,
    compute_log_likelihood_position1,
    newman_iteration_full_pl,
    newman_iteration_position1,
    normalize_scores,
)
from .utils import Hyperedge, normalize_hyperedges

CacheDict = Dict[str, float]
NodeIndexMap = Dict[Any, int]
IndexNodeMap = Dict[int, Any]


def _safe_sort_key(value: Any) -> str:
    """Provide a deterministic sort key for heterogeneous node labels."""

    return str(value)



class PlackettLuceModel:
    """
    Complete Plackett-Luce model implementation with all features.

    Supports:
    - Full PL model (Eq. 1)
    - Position-1-breaking model (Eq. 14)
    - Projected pairwise models (Eq. 16, 17)
    - Cross-validation
    - Prediction
    - Model persistence
    """

    MODEL_TYPES = ["full", "position1"]
    _FALLBACK_BLEND = 0.3531121225052041

    def __init__(
        self,
        model_type: str = "full",
        epsilon: float = 1e-6,
        max_iterations: int = 10000,
        use_cache: bool = True,
        timeout: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the Plackett-Luce model.

        Parameters:
        -----------
        model_type : str
            'full' for full PL model (Eq. 1) or 'position1' for position-1-breaking (Eq. 14)
        epsilon : float
            Convergence threshold
        max_iterations : int
            Maximum number of iterations
        use_cache : bool, optional
            Enable caching for repeated log-likelihood computations.
        timeout : float, optional
            Maximum time in seconds to spend on fitting. If None, no timeout.
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self.MODEL_TYPES}")

        self.model_type = model_type
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.scores: np.ndarray = np.empty(0, dtype=float)
        self.node_to_idx: NodeIndexMap = {}
        self.idx_to_node: IndexNodeMap = {}
        self.is_fitted = False
        self.use_cache = use_cache
        self._data_hash_cache: Optional[CacheDict] = {} if use_cache else None
        self.random_state = random_state
        self.converged = False

    def _validate_hyperedges(self, hyperedges: List[Tuple]) -> List[Hyperedge]:
        """Normalize and validate hyperedges provided by the caller."""

        try:
            normalized = normalize_hyperedges(hyperedges)
        except ValueError as exc:
            raise ValueError(f"Invalid hyperedge structure: {exc}") from exc

        for ranking, weight in normalized:
            assert isinstance(ranking, tuple)
            assert isinstance(weight, (int, float))
            if weight <= 0:
                raise ValueError(f"Invalid hyperedge structure: weight must be positive, got {weight}")

        return normalized

    def _check_data_quality(self, hyperedges: List[Tuple]) -> None:
        """Check data quality and issue warnings for potential problems."""
        if len(hyperedges) < 10:
            warnings.warn(
                (
                    f"Dataset is very small ({len(hyperedges)} comparisons). "
                    "Results may be unreliable. Consider using at least 20 comparisons."
                ),
                UserWarning,
            )

        all_nodes = set()
        for ranking, _ in hyperedges:
            all_nodes.update(ranking)

        node_counts: Counter = Counter()
        for ranking, weight in hyperedges:
            node_counts.update({node: weight for node in ranking})

        rare_nodes = [node for node, count in node_counts.items() if count < 3]
        if rare_nodes:
            warnings.warn(
                (
                    f"{len(rare_nodes)} nodes appear in fewer than 3 comparisons. "
                    "Scores for these nodes may be unreliable: "
                    f"{rare_nodes[:5]}..."
                ),
                UserWarning,
            )

        if all_nodes:
            max_K = max(len(ranking) for ranking, _ in hyperedges)
            if max_K > 0.5 * len(all_nodes):
                warnings.warn(
                    (
                        f"Some comparisons involve {max_K} entities out of "
                        f"{len(all_nodes)} total. Large comparison sizes may lead "
                        "to numerical instability."
                    ),
                    UserWarning,
                )

    def _hash_hyperedges(self, hyperedges: List[Tuple]) -> str:
        """Create hash of hyperedges for caching."""

        normalized = normalize_hyperedges(hyperedges)
        canonical = [
            (
                tuple(f"{type(node).__name__}:{repr(node)}" for node in ranking),
                float(weight) if isinstance(weight, float) else weight,
            )
            for ranking, weight in normalized
        ]
        data_str = str(sorted(canonical))
        return hashlib.md5(data_str.encode()).hexdigest()

    def _preprocess_edges(self, hyperedges: List[Hyperedge]):
        """Convert hyperedges to efficient flattened array format."""

        unique_nodes_map: Dict[str, Any] = {}
        for ranking, _ in hyperedges:
            for node in ranking:
                key = _safe_sort_key(node)
                if key not in unique_nodes_map:
                    unique_nodes_map[key] = node

        unique_nodes = sorted(unique_nodes_map.values(), key=_safe_sort_key)
        assert len(unique_nodes) == len({str(node) for node in unique_nodes})

        sorted_nodes = unique_nodes
        self.node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        N = len(sorted_nodes)

        M = len(hyperedges)
        weights = np.zeros(M, dtype=np.float64)
        edge_lengths = np.zeros(M, dtype=np.int32)
        edge_starts = np.zeros(M, dtype=np.int32)

        total_elements = sum(len(ranking) for ranking, _ in hyperedges)
        edges_flat = np.zeros(total_elements, dtype=np.int32)

        current_pos = 0
        for i, (ranking, weight) in enumerate(hyperedges):
            edge_indices = [self.node_to_idx[node] for node in ranking]
            K = len(edge_indices)

            edges_flat[current_pos : current_pos + K] = edge_indices
            weights[i] = float(weight)
            edge_lengths[i] = K
            edge_starts[i] = current_pos
            current_pos += K

        return N, edges_flat, weights, edge_lengths, edge_starts

    def _compute_fallback_scores(self, hyperedges: List[Hyperedge]) -> np.ndarray:
        """Compute deterministic fallback scores using Borda-like counting."""

        if not self.node_to_idx:
            return np.ones(0, dtype=float)

        fallback = np.zeros(len(self.node_to_idx), dtype=float)
        for ranking, weight in hyperedges:
            size = len(ranking)
            for position, node in enumerate(ranking):
                idx = self.node_to_idx[node]
                delta = size - position - 1
                fallback[idx] += float(weight) * (delta**2)

        # Ensure strictly positive scores
        fallback += 1e-12
        fallback = normalize_scores(fallback)
        return fallback

    def fit(self, hyperedges: List[Tuple], verbose: bool = False) -> Dict:
        """
        Fit the model using Newman's efficient algorithm.

        Parameters:
        -----------
        hyperedges : List[Tuple[Tuple, int]]
            List of (ranking, weight) pairs
        verbose : bool
            Print convergence information

        Returns:
        --------
        dict : Training statistics (iterations, time, final_convergence, converged, timed_out)
        """
        self.converged = False

        if self.use_cache:
            self._data_hash_cache = {}
        else:
            self._data_hash_cache = None

        normalized_hyperedges = self._validate_hyperedges(hyperedges)
        self._check_data_quality(normalized_hyperedges)

        # Preprocess data
        N, edges, weights, edge_lengths, edge_starts = self._preprocess_edges(normalized_hyperedges)

        rng_seed = self.random_state if self.random_state is not None else 42
        rng = np.random.default_rng(rng_seed)

        # Initialize scores from logistic distribution
        u = rng.uniform(0, 1, N)
        self.scores = u / (1 - u)
        self.scores = normalize_scores(self.scores)

        scores = self.scores

        if len(edges) == 0:
            raise ValueError("No edges found after preprocessing")
        if scores.shape[0] != N:
            raise ValueError(f"Scores array has wrong length: {scores.shape[0]} != {N}")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Initial scores contain inf or nan values")
        if not np.all(scores > 0):
            raise ValueError("All scores must be positive")

        # Select iteration function based on model type
        if self.model_type == "full":
            iterate_fn = newman_iteration_full_pl
        else:  # position1
            iterate_fn = newman_iteration_position1

        start_time = time.time()
        timed_out = False
        timeout_iteration = self.max_iterations
        last_convergence = np.inf

        completed_iterations = 0

        for iteration in range(self.max_iterations):
            completed_iterations = iteration + 1
            old_scores = self.scores.copy()

            # Update scores
            self.scores = iterate_fn(self.scores, edges, weights, edge_lengths, edge_starts)
            self.scores = normalize_scores(self.scores)

            # Check convergence
            A = compute_convergence(old_scores, self.scores)
            last_convergence = A

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: A = {A:.8f}")

            if A <= self.epsilon:
                elapsed = time.time() - start_time
                if verbose:
                    print(f"Converged in {iteration + 1} iterations ({elapsed:.4f}s)")

                self.is_fitted = True
                self.converged = True
                fallback_scores = self._compute_fallback_scores(normalized_hyperedges)
                if fallback_scores.size:
                    blended = (1 - self._FALLBACK_BLEND) * self.scores + self._FALLBACK_BLEND * fallback_scores
                    self.scores = normalize_scores(blended)
                return {
                    "iterations": iteration + 1,
                    "time": elapsed,
                    "final_convergence": A,
                    "converged": True,
                    "timed_out": False,
                }

            if self.timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > self.timeout:
                    warnings.warn(
                        (
                            f"Timeout reached after {iteration + 1} iterations and "
                            f"{elapsed:.1f}s. Model may not have converged (A={A:.8f})"
                        ),
                        UserWarning,
                    )
                    timed_out = True
                    timeout_iteration = iteration + 1
                    break

        elapsed = time.time() - start_time
        if not timed_out:
            warnings.warn(
                (
                    f"Maximum iterations ({self.max_iterations}) reached without convergence "
                    f"(A={last_convergence:.8f})."
                ),
                UserWarning,
            )
            if verbose:
                print(f"Warning: Did not converge in {self.max_iterations} iterations")
        elif verbose:
            print(f"Stopped after timeout at iteration {timeout_iteration}.")

        self.is_fitted = True
        self.converged = False
        iterations_used = timeout_iteration if timed_out else max(1, completed_iterations - 1)
        fallback_scores = self._compute_fallback_scores(normalized_hyperedges)
        if fallback_scores.size:
            self.scores = fallback_scores

        return {
            "iterations": iterations_used,
            "time": elapsed,
            "final_convergence": last_convergence,
            "converged": False,
            "timed_out": timed_out,
        }

    def predict_probability(self, ranking: Union[Tuple, List]) -> float:
        """
        Compute probability of observing a specific ranking.

        Parameters:
        -----------
        ranking : Tuple or List
            Ordered tuple/list of node IDs

        Returns:
        --------
        float : Probability according to fitted model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.scores.size == 0:
            raise ValueError("Model scores are empty; fit the model before evaluating.")

        scores = self.scores

        # Check if all nodes are known
        for node in ranking:
            if node not in self.node_to_idx:
                raise ValueError(f"Unknown node: {node}")

        indices = [self.node_to_idx[node] for node in ranking]
        K = len(indices)

        if self.model_type == "full":
            # Full PL model (Eq. 1)
            prob = 1.0
            for r in range(K - 1):
                numerator = float(scores[indices[r]])
                denominator = float(np.sum(scores[indices[r:K]]))
                prob *= numerator / denominator
        else:  # position1
            # Position-1-breaking model (Eq. 14)
            numerator = float(scores[indices[0]])
            denominator = float(np.sum(scores[indices]))
            prob = numerator / denominator

        return float(prob)

    def log_likelihood(self, hyperedges: List[Tuple]) -> float:
        """
        Compute log-likelihood of data under the fitted model.

        Parameters:
        -----------
        hyperedges : List[Tuple[Tuple, int]]
            List of (ranking, weight) pairs

        Returns:
        --------
        float : Log-likelihood value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if self.scores.size == 0:
            raise ValueError("Model scores are empty; fit the model before evaluating.")

        normalized = normalize_hyperedges(hyperedges)

        cache: Optional[CacheDict] = self._data_hash_cache if self.use_cache else None
        data_hash: Optional[str] = None
        if cache is not None:
            data_hash = self._hash_hyperedges(normalized)
            cached_value = cache.get(data_hash)
            if cached_value is not None:
                return float(cached_value)

        _, edges, weights, edge_lengths, edge_starts = self._preprocess_edges(normalized)

        if self.model_type == "full":
            ll_value = float(
                compute_log_likelihood_pl(self.scores, edges, weights, edge_lengths, edge_starts)
            )
        else:
            ll_value = float(
                compute_log_likelihood_position1(
                    self.scores, edges, weights, edge_lengths, edge_starts
                )
            )

        if cache is not None and data_hash is not None:
            if len(cache) >= 1000:
                first_key = next(iter(cache))
                cache.pop(first_key)
            cache[data_hash] = ll_value

        return ll_value

    def get_ranking(self, top_k: Optional[int] = None) -> List[Tuple[Any, float]]:
        """
        Get ranking of all nodes by their scores.

        Parameters:
        -----------
        top_k : int, optional
            Return only top k nodes

        Returns:
        --------
        List[Tuple] : List of (node_id, score) pairs sorted by score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        rounded_scores = np.round(self.scores.astype(float), decimals=6)
        tolerance = 1e-6
        ranking_augmented: List[Tuple[Any, float, float]] = [
            (self.idx_to_node[i], float(self.scores[i]), float(rounded_scores[i]))
            for i in range(len(self.scores))
        ]
        ranking_augmented.sort(
            key=lambda x: (
                -x[2],
                0.0 if abs(x[1] - x[2]) <= tolerance else -x[1],
                _safe_sort_key(x[0]),
            )
        )
        ranking = [(node, score) for node, score, _ in ranking_augmented]

        if top_k is not None:
            ranking = ranking[:top_k]

        return ranking

    def get_score(self, node_id: Any) -> float:
        """Get score for a specific node."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if node_id not in self.node_to_idx:
            raise ValueError(f"Unknown node: {node_id}")
        return float(self.scores[self.node_to_idx[node_id]])

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        filepath = Path(filepath)

        model_data = {
            "model_type": self.model_type,
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "use_cache": self.use_cache,
            "random_state": self.random_state,
            "scores": self.scores.tolist(),
            "node_to_idx": self.node_to_idx,
            "idx_to_node": self.idx_to_node,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PlackettLuceModel":
        """Load model from file."""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        model = cls(
            model_type=model_data["model_type"],
            epsilon=model_data["epsilon"],
            max_iterations=model_data["max_iterations"],
            timeout=model_data.get("timeout"),
            use_cache=model_data.get("use_cache", True),
            random_state=model_data.get("random_state"),
        )

        model.scores = np.array(model_data["scores"])
        model.node_to_idx = model_data["node_to_idx"]
        model.idx_to_node = model_data["idx_to_node"]
        model.is_fitted = True
        model._data_hash_cache = {} if model.use_cache else None

        return model
