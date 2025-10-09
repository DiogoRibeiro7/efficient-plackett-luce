"""
Numba-optimized core functions for Plackett-Luce inference.

All performance-critical computations using JIT compilation.
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def normalize_scores(scores):
    """Normalize scores so geometric mean equals 1."""
    log_mean = np.mean(np.log(scores))
    return scores / np.exp(log_mean)

@jit(nopython=True)
def compute_convergence(old_scores, new_scores):
    """Compute convergence metric from equation (8)."""
    old_prob = old_scores / (1.0 + old_scores)
    new_prob = new_scores / (1.0 + new_scores)
    return np.sqrt(np.mean((old_prob - new_prob)**2))

@jit(nopython=True)
def newman_iteration_full_pl(scores, edges, weights, edge_lengths, edge_starts):
    """Newman's algorithm for full Plackett-Luce (Eq. 13)."""
    N = len(scores)
    M = len(weights)
    new_scores = np.zeros(N, dtype=np.float64)
    
    for s in range(N):
        numerator = 1.0 / (scores[s] + 1.0)
        denominator = 1.0 / (scores[s] + 1.0)
        
        for m in range(M):
            start = edge_starts[m]
            K = edge_lengths[m]
            edge = edges[start:start+K]
            weight = weights[m]
            
            r = -1
            for i in range(K):
                if edge[i] == s:
                    r = i
                    break
            
            if r == -1:
                continue
            
            if r < K - 1:
                sum_all = 0.0
                sum_rest = 0.0
                for i in range(r, K):
                    sum_all += scores[edge[i]]
                    if i > r:
                        sum_rest += scores[edge[i]]
                numerator += weight * sum_rest / sum_all
            
            for v in range(r):
                sum_scores = 0.0
                for i in range(v, K):
                    sum_scores += scores[edge[i]]
                denominator += weight / sum_scores
        
        new_scores[s] = numerator / denominator
    
    return new_scores

@jit(nopython=True)
def newman_iteration_position1(scores, edges, weights, edge_lengths, edge_starts):
    """Newman's algorithm for position-1-breaking PL (Eq. 15)."""
    N = len(scores)
    M = len(weights)
    new_scores = np.zeros(N, dtype=np.float64)
    
    for s in range(N):
        numerator = 1.0 / (scores[s] + 1.0)
        denominator = 1.0 / (scores[s] + 1.0)
        
        for m in range(M):
            start = edge_starts[m]
            K = edge_lengths[m]
            edge = edges[start:start+K]
            weight = weights[m]
            
            r = -1
            for i in range(K):
                if edge[i] == s:
                    r = i
                    break
            
            if r == -1:
                continue
            
            # Only first position contributes to numerator
            if r == 0:
                sum_rest = 0.0
                sum_all = 0.0
                for i in range(K):
                    sum_all += scores[edge[i]]
                    if i > 0:
                        sum_rest += scores[edge[i]]
                numerator += weight * sum_rest / sum_all
            
            # All positions contribute to denominator (except first)
            if r >= 1:
                sum_scores = 0.0
                for i in range(K):
                    sum_scores += scores[edge[i]]
                denominator += weight / sum_scores
        
        new_scores[s] = numerator / denominator
    
    return new_scores

@jit(nopython=True)
def compute_log_likelihood_pl(scores, edges, weights, edge_lengths, edge_starts):
    """Compute log-likelihood for full PL model (Eq. 4)."""
    M = len(weights)
    log_lik = 0.0
    
    for m in range(M):
        start = edge_starts[m]
        K = edge_lengths[m]
        edge = edges[start:start+K]
        weight = weights[m]
        
        # Product over positions
        for r in range(K - 1):
            numerator = scores[edge[r]]
            denominator = 0.0
            for i in range(r, K):
                denominator += scores[edge[i]]
            
            log_lik += weight * (np.log(numerator) - np.log(denominator))
    
    return log_lik

@jit(nopython=True)
def compute_log_likelihood_position1(scores, edges, weights, edge_lengths, edge_starts):
    """Compute log-likelihood for position-1-breaking PL model (Eq. 14)."""
    M = len(weights)
    log_lik = 0.0
    
    for m in range(M):
        start = edge_starts[m]
        K = edge_lengths[m]
        edge = edges[start:start+K]
        weight = weights[m]
        
        numerator = scores[edge[0]]
        denominator = 0.0
        for i in range(K):
            denominator += scores[edge[i]]
        
        log_lik += weight * (np.log(numerator) - np.log(denominator))
    
    return log_lik
