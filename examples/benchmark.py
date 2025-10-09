"""
Performance benchmarking examples.

Compares:
- Zermelo vs Newman algorithms
- Different dataset sizes
- Convergence speed
"""

from plackett_luce import PlackettLuceModel
from plackett_luce.utils import generate_synthetic_rankings
import time
import numpy as np

def benchmark_convergence(N, M, K_min, K_max, n_runs=5):
    """Benchmark model on synthetic data."""
    times = []
    iterations = []
    
    for run in range(n_runs):
        data = generate_synthetic_rankings(N, M, K_min, K_max, seed=run)
        
        model = PlackettLuceModel(model_type='full')
        start = time.time()
        stats = model.fit(data, verbose=False)
        elapsed = time.time() - start
        
        times.append(elapsed)
        iterations.append(stats['iterations'])
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_iters': np.mean(iterations),
        'std_iters': np.std(iterations),
        'throughput': M / np.mean(times)  # comparisons per second
    }

def main():
    print("=" * 70)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    # Benchmark 1: Scaling with dataset size
    print("\n1. Scaling with Dataset Size")
    print("-" * 70)
    
    test_cases = [
        (50, 100, "Tiny"),
        (100, 500, "Small"),
        (200, 1000, "Medium"),
        (500, 2500, "Large"),
        (1000, 5000, "Very Large"),
    ]
    
    print(f"\n{'Size':<12} {'N':>6} {'M':>7} {'Time':>10} {'Iters':>8} {'Throughput':>12}")
    print("-" * 70)
    
    for N, M, label in test_cases:
        print(f"{label:<12} {N:>6} {M:>7}", end=" ", flush=True)
        result = benchmark_convergence(N, M, K_min=2, K_max=5, n_runs=3)
        print(f"{result['mean_time']:>9.3f}s {result['mean_iters']:>7.1f} {result['throughput']:>10.0f} cmp/s")
    
    # Benchmark 2: Effect of comparison size
    print("\n2. Effect of Comparison Size (K)")
    print("-" * 70)
    
    N, M = 200, 1000
    k_ranges = [
        (2, 2, "Pairwise only"),
        (2, 4, "Small groups"),
        (2, 8, "Medium groups"),
        (2, 15, "Large groups"),
    ]
    
    print(f"\n{'K range':<20} {'Time':>10} {'Iters':>8}")
    print("-" * 45)
    
    for K_min, K_max, label in k_ranges:
        result = benchmark_convergence(N, M, K_min, K_max, n_runs=3)
        print(f"{label:<20} {result['mean_time']:>9.3f}s {result['mean_iters']:>7.1f}")
    
    # Benchmark 3: Convergence behavior
    print("\n3. Convergence Behavior")
    print("-" * 70)
    
    data = generate_synthetic_rankings(N=100, M=500, K_min=2, K_max=5, seed=42)
    
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    
    print(f"\n{'Epsilon':<12} {'Iterations':>12} {'Time':>10}")
    print("-" * 40)
    
    for eps in epsilons:
        model = PlackettLuceModel(model_type='full', epsilon=eps)
        start = time.time()
        stats = model.fit(data, verbose=False)
        elapsed = time.time() - start
        print(f"{eps:<12.0e} {stats['iterations']:>12} {elapsed:>9.3f}s")
    
    # Benchmark 4: Model type comparison
    print("\n4. Model Type Comparison")
    print("-" * 70)
    
    data = generate_synthetic_rankings(N=200, M=1000, K_min=2, K_max=8, seed=42)
    
    model_types = ['full', 'position1']
    
    print(f"\n{'Model Type':<15} {'Time':>10} {'Iters':>8} {'Throughput':>12}")
    print("-" * 50)
    
    for model_type in model_types:
        times = []
        iters = []
        
        for _ in range(3):
            model = PlackettLuceModel(model_type=model_type)
            start = time.time()
            stats = model.fit(data, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)
            iters.append(stats['iterations'])
        
        mean_time = np.mean(times)
        throughput = len(data) / mean_time
        print(f"{model_type:<15} {mean_time:>9.3f}s {np.mean(iters):>7.1f} {throughput:>10.0f} cmp/s")
    
    # Benchmark 5: Memory usage estimate
    print("\n5. Memory Usage Estimation")
    print("-" * 70)
    
    test_sizes = [
        (100, 1000),
        (500, 5000),
        (1000, 10000),
        (5000, 50000),
    ]
    
    print(f"\n{'N nodes':<10} {'M comparisons':<15} {'Est. Memory':<12}")
    print("-" * 40)
    
    for N, M in test_sizes:
        # Rough estimate: scores (N floats) + edges data (M * avg_K * 2)
        avg_K = 5
        memory_mb = (N * 8 + M * avg_K * 8 * 2) / (1024 * 1024)
        print(f"{N:<10} {M:<15} {memory_mb:>10.2f} MB")
    
    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
