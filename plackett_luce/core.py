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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .optimization import (
    compute_convergence,
    compute_log_likelihood_pl,
    compute_log_likelihood_position1,
    newman_iteration_full_pl,
    newman_iteration_position1,
    normalize_scores,
)


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

    def __init__(
        self,
        model_type: str = "full",
        epsilon: float = 1e-6,
        max_iterations: int = 10000,
        use_cache: bool = True,
        timeout: Optional[float] = None,
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
        self.scores = None
        self.node_to_idx = None
        self.idx_to_node = None
        self.is_fitted = False
        self.use_cache = use_cache
        self._data_hash_cache = {} if use_cache else None

    def _validate_hyperedges(self, hyperedges: List[Tuple]) -> None:
        """Validate input hyperedges format."""
        if not hyperedges:
            raise ValueError("hyperedges cannot be empty")

        for i, item in enumerate(hyperedges):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    f"Each hyperedge must be (ranking, weight) tuple, got {item} at index {i}"
                )

            ranking, weight = item
            if not isinstance(ranking, (tuple, list)) or len(ranking) < 2:
                raise ValueError(
                    f"Ranking must be tuple/list with at least 2 elements at index {i}"
                )

            if len(ranking) != len(set(ranking)):
                duplicates = list(
                    dict.fromkeys(node for node in ranking if ranking.count(node) > 1)
                )
                raise ValueError(
                    f"Ranking contains duplicate nodes {duplicates} at index {i}. "
                    f"Each entity can appear only once per ranking."
                )

            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(f"Weight must be positive number at index {i}")

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
        data_str = str(sorted(hyperedges))
        return hashlib.md5(data_str.encode()).hexdigest()

    def _preprocess_edges(self, hyperedges: List[Tuple]):
        """Convert hyperedges to efficient flattened array format."""
        # Build node mapping
        unique_nodes = set()
        for edge, _ in hyperedges:
            unique_nodes.update(edge)

        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        N = len(unique_nodes)

        # Flatten edges into arrays
        M = len(hyperedges)
        weights = np.zeros(M, dtype=np.float64)
        edge_lengths = np.zeros(M, dtype=np.int32)
        edge_starts = np.zeros(M, dtype=np.int32)

        total_elements = sum(len(edge) for edge, _ in hyperedges)
        edges_flat = np.zeros(total_elements, dtype=np.int32)

        current_pos = 0
        for i, (edge, weight) in enumerate(hyperedges):
            edge_indices = [self.node_to_idx[node] for node in edge]
            K = len(edge_indices)

            edges_flat[current_pos : current_pos + K] = edge_indices
            weights[i] = weight
            edge_lengths[i] = K
            edge_starts[i] = current_pos
            current_pos += K

        return N, edges_flat, weights, edge_lengths, edge_starts

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
        if self.use_cache and self._data_hash_cache is not None:
            self._data_hash_cache.clear()
        self._validate_hyperedges(hyperedges)
        self._check_data_quality(hyperedges)

        # Preprocess data
        N, edges, weights, edge_lengths, edge_starts = self._preprocess_edges(hyperedges)

        # Initialize scores from logistic distribution
        u = np.random.uniform(0, 1, N)
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

        for iteration in range(self.max_iterations):
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
                return {
                    "iterations": iteration + 1,
                    "time": elapsed,
                    "final_convergence": A,
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
        if verbose:
            print(f"Warning: Did not converge in {self.max_iterations} iterations")

        self.is_fitted = True
        iterations_used = timeout_iteration if timed_out else self.max_iterations
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
                numerator = self.scores[indices[r]]
                denominator = sum(self.scores[indices[i]] for i in range(r, K))
                prob *= numerator / denominator
        else:  # position1
            # Position-1-breaking model (Eq. 14)
            numerator = self.scores[indices[0]]
            denominator = sum(self.scores[indices[i]] for i in range(K))
            prob = numerator / denominator

        return prob

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

        cache_enabled = self.use_cache and self._data_hash_cache is not None
        data_hash = None
        if cache_enabled:
            data_hash = self._hash_hyperedges(hyperedges)
            cached_value = self._data_hash_cache.get(data_hash)
            if cached_value is not None:
                return cached_value

        _, edges, weights, edge_lengths, edge_starts = self._preprocess_edges(hyperedges)

        if self.model_type == "full":
            ll = compute_log_likelihood_pl(self.scores, edges, weights, edge_lengths, edge_starts)
        else:
            ll = compute_log_likelihood_position1(
                self.scores, edges, weights, edge_lengths, edge_starts
            )

        if cache_enabled and data_hash is not None:
            if len(self._data_hash_cache) >= 1000:
                # Remove an arbitrary cached item to control memory usage
                self._data_hash_cache.pop(next(iter(self._data_hash_cache)))
            self._data_hash_cache[data_hash] = ll

        return ll

    def get_ranking(self, top_k: Optional[int] = None) -> List[Tuple]:
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

        ranking = [(self.idx_to_node[i], self.scores[i]) for i in range(len(self.scores))]
        ranking.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranking = ranking[:top_k]

        return ranking

    def get_score(self, node_id) -> float:
        """Get score for a specific node."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        if node_id not in self.node_to_idx:
            raise ValueError(f"Unknown node: {node_id}")
        return self.scores[self.node_to_idx[node_id]]

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
        )

        model.scores = np.array(model_data["scores"])
        model.node_to_idx = model_data["node_to_idx"]
        model.idx_to_node = model_data["idx_to_node"]
        model.is_fitted = True

        return model
