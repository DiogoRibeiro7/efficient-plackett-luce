"""
Cross-validation and model selection examples.

Demonstrates:
- K-fold cross-validation
- Model comparison
- Train/test splitting strategies
"""

from plackett_luce import PlackettLuceModel, cross_validate
from plackett_luce.validation import train_test_split, stratified_split_by_size
from plackett_luce.utils import generate_synthetic_rankings
import numpy as np

def main():
    print("=" * 70)
    print("CROSS-VALIDATION EXAMPLES")
    print("=" * 70)
    
    # Generate synthetic data
    print("\nGenerating synthetic tournament data...")
    np.random.seed(42)
    data = generate_synthetic_rankings(N=50, M=500, K_min=2, K_max=5)
    print(f"Created {len(data)} comparisons with {len(set(e for r,_ in data for e in r))} entities")
    
    # Example 1: Simple train/test split
    print("\n1. Simple Train/Test Split")
    print("-" * 70)
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train size: {len(train)} ({len(train)/len(data)*100:.1f}%)")
    print(f"Test size:  {len(test)} ({len(test)/len(data)*100:.1f}%)")
    
    model = PlackettLuceModel(model_type='full')
    model.fit(train, verbose=False)
    
    train_ll = model.log_likelihood(train)
    test_ll = model.log_likelihood(test)
    
    print(f"\nLog-likelihoods:")
    print(f"  Train: {train_ll:.4f}")
    print(f"  Test:  {test_ll:.4f}")
    print(f"  Generalization gap: {train_ll - test_ll:.4f}")
    
    # Example 2: Stratified split
    print("\n2. Stratified Split (by comparison size)")
    print("-" * 70)
    
    train_strat, test_strat = stratified_split_by_size(data, test_size=0.2, random_state=42)
    
    def get_size_distribution(dataset):
        from collections import Counter
        sizes = [len(ranking) for ranking, _ in dataset]
        return Counter(sizes)
    
    print("\nComparison size distribution:")
    print(f"  Original: {dict(get_size_distribution(data))}")
    print(f"  Train:    {dict(get_size_distribution(train_strat))}")
    print(f"  Test:     {dict(get_size_distribution(test_strat))}")
    
    # Example 3: K-fold cross-validation
    print("\n3. K-Fold Cross-Validation")
    print("-" * 70)
    
    cv_results = cross_validate(
        data,
        model_type='full',
        n_splits=5,
        compare_projected=True,
        random_state=42,
        verbose=True
    )
    
    print(f"\nCross-Validation Results:")
    print(f"  Full PL Model:")
    print(f"    Mean Log-Likelihood: {cv_results['pl_mean']:.4f}")
    print(f"    Std Dev:             {cv_results['pl_std']:.4f}")
    print(f"  Projected Model:")
    print(f"    Mean Log-Likelihood: {cv_results['projected_mean']:.4f}")
    print(f"    Std Dev:             {cv_results['projected_std']:.4f}")
    print(f"\n  Winner: {cv_results['winner']}")
    print(f"  Improvement: {cv_results['improvement']:.2f}%")
    
    # Example 4: Model comparison
    print("\n4. Comparing Different Model Types")
    print("-" * 70)
    
    models = {
        'Full PL': PlackettLuceModel(model_type='full'),
        'Position-1': PlackettLuceModel(model_type='position1'),
    }
    
    results_comparison = {}
    
    for name, model in models.items():
        cv_result = cross_validate(
            data,
            model_type=model.model_type,
            n_splits=5,
            compare_projected=False,
            random_state=42,
            verbose=False
        )
        results_comparison[name] = cv_result
    
    print("\nModel Comparison (5-fold CV):")
    print(f"{'Model':<15} {'Mean LL':<12} {'Std Dev':<10}")
    print("-" * 40)
    for name, result in results_comparison.items():
        print(f"{name:<15} {result['pl_mean']:>10.4f}  {result['pl_std']:>8.4f}")
    
    # Find best model
    best_model = max(results_comparison.items(), key=lambda x: x[1]['pl_mean'])
    print(f"\n✓ Best model: {best_model[0]}")
    
    # Example 5: Learning curves
    print("\n5. Learning Curves (vary training size)")
    print("-" * 70)
    
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    learning_curve = []
    
    for train_size in train_sizes:
        train, test = train_test_split(data, test_size=1-train_size, random_state=42)
        model = PlackettLuceModel(model_type='full')
        model.fit(train, verbose=False)
        test_ll = model.log_likelihood(test)
        learning_curve.append((len(train), test_ll))
    
    print("\nLearning Curve:")
    print(f"{'Train Size':<12} {'Test Log-Likelihood':<20}")
    print("-" * 35)
    for n_train, ll in learning_curve:
        print(f"{n_train:<12} {ll:>18.4f}")

if __name__ == "__main__":
    main()
