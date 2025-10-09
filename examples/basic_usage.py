"""
Basic usage examples for efficient-plackett-luce.

This script demonstrates the fundamental operations:
- Loading data
- Fitting models
- Getting rankings
- Making predictions
"""

from plackett_luce import PlackettLuceModel

def main():
    print("=" * 70)
    print("BASIC USAGE EXAMPLES")
    print("=" * 70)
    
    # Example 1: Simple tournament
    print("\n1. Simple Tournament Ranking")
    print("-" * 70)
    
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    results = [
        (("TeamA", "TeamB", "TeamC"), 1),
        (("TeamB", "TeamC", "TeamD"), 1),
        (("TeamA", "TeamD"), 2),
        (("TeamC", "TeamB", "TeamA"), 1),
    ]
    
    model = PlackettLuceModel(model_type='full')
    stats = model.fit(results, verbose=True)
    
    print(f"\nTraining completed in {stats['iterations']} iterations")
    print(f"Time: {stats['time']:.4f} seconds")
    
    print("\nFinal Rankings:")
    for rank, (team, score) in enumerate(model.get_ranking(), 1):
        print(f"  {rank}. {team:10s} Score: {score:.4f}")
    
    # Example 2: Predictions
    print("\n2. Making Predictions")
    print("-" * 70)
    
    test_rankings = [
        ("TeamA", "TeamB", "TeamC"),
        ("TeamB", "TeamA", "TeamC"),
        ("TeamC", "TeamB", "TeamA"),
    ]
    
    print("\nProbabilities of different outcomes:")
    for ranking in test_rankings:
        prob = model.predict_probability(ranking)
        print(f"  P{ranking} = {prob:.6f}")
    
    # Example 3: Model persistence
    print("\n3. Saving and Loading Models")
    print("-" * 70)
    
    # Save
    model.save("tournament_model.pkl")
    print("✓ Model saved to 'tournament_model.pkl'")
    
    # Load
    loaded_model = PlackettLuceModel.load("tournament_model.pkl")
    print("✓ Model loaded successfully")
    
    # Verify
    original_ranking = model.get_ranking()
    loaded_ranking = loaded_model.get_ranking()
    
    print("\nVerification:")
    print(f"  Original top team: {original_ranking[0][0]}")
    print(f"  Loaded top team:   {loaded_ranking[0][0]}")
    print(f"  Match: {'✓' if original_ranking == loaded_ranking else '✗'}")
    
    # Cleanup
    import os
    os.remove("tournament_model.pkl")
    print("\n✓ Cleaned up temporary files")

if __name__ == "__main__":
    main()
