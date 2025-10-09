"""
FIFA World Cup ranking example.

Demonstrates how to:
- Model tournament data with group stages
- Handle repeated matchups
- Compare different model types
"""

from plackett_luce import PlackettLuceModel, ProjectedPlackettLuce

def main():
    print("=" * 70)
    print("FIFA WORLD CUP RANKING EXAMPLE")
    print("=" * 70)
    
    # Simplified World Cup results (fictional for demonstration)
    print("\nDataset: Simplified World Cup Results")
    print("-" * 70)
    
    results = [
        # Group A
        (("Brazil", "Croatia", "Mexico", "Cameroon"), 1),
        # Group B
        (("Spain", "Netherlands", "Chile", "Australia"), 1),
        # Group C
        (("Germany", "Portugal", "USA", "Ghana"), 1),
        # Group D
        (("Argentina", "Belgium", "Switzerland", "Bosnia"), 1),
        # Round of 16
        (("Brazil", "Chile"), 1),
        (("Colombia", "Uruguay"), 1),
        (("Netherlands", "Mexico"), 1),
        (("Germany", "Algeria"), 1),
        (("France", "Nigeria"), 1),
        (("Argentina", "Switzerland"), 1),
        (("Belgium", "USA"), 1),
        (("Costa Rica", "Greece"), 1),
        # Quarter-finals
        (("Brazil", "Colombia"), 1),
        (("Germany", "France"), 1),
        (("Netherlands", "Costa Rica"), 1),
        (("Argentina", "Belgium"), 1),
        # Semi-finals
        (("Germany", "Brazil"), 1),
        (("Argentina", "Netherlands"), 1),
        # Third place
        (("Netherlands", "Brazil"), 1),
        # Final
        (("Germany", "Argentina"), 1),
    ]
    
    n_teams = len(set(team for ranking, _ in results for team in ranking))
    n_matches = sum(weight for _, weight in results)
    print(f"Total teams: {n_teams}")
    print(f"Total comparisons: {n_matches}")
    
    # Model 1: Full Plackett-Luce
    print("\n1. Full Plackett-Luce Model")
    print("-" * 70)
    
    model_full = PlackettLuceModel(model_type='full')
    model_full.fit(results, verbose=True)
    
    print("\nTop 10 Teams (Full PL):")
    for rank, (team, score) in enumerate(model_full.get_ranking(top_k=10), 1):
        print(f"  {rank:2d}. {team:15s} {score:.4f}")
    
    # Model 2: Position-1-breaking
    print("\n2. Position-1-Breaking Model")
    print("-" * 70)
    
    model_pos1 = PlackettLuceModel(model_type='position1')
    model_pos1.fit(results, verbose=True)
    
    print("\nTop 10 Teams (Position-1):")
    for rank, (team, score) in enumerate(model_pos1.get_ranking(top_k=10), 1):
        print(f"  {rank:2d}. {team:15s} {score:.4f}")
    
    # Model 3: Projected Pairwise
    print("\n3. Projected Pairwise Model")
    print("-" * 70)
    
    model_proj = ProjectedPlackettLuce(model_type='full')
    model_proj.fit(results, verbose=True)
    
    print("\nTop 10 Teams (Projected):")
    for rank, (team, score) in enumerate(model_proj.get_ranking(top_k=10), 1):
        print(f"  {rank:2d}. {team:15s} {score:.4f}")
    
    # Compare rankings
    print("\n4. Ranking Comparison")
    print("-" * 70)
    
    def get_top_5(model):
        return [team for team, _ in model.get_ranking(top_k=5)]
    
    top5_full = get_top_5(model_full)
    top5_pos1 = get_top_5(model_pos1)
    top5_proj = get_top_5(model_proj)
    
    print("\nTop 5 comparison:")
    print(f"  Full PL:    {', '.join(top5_full)}")
    print(f"  Position-1: {', '.join(top5_pos1)}")
    print(f"  Projected:  {', '.join(top5_proj)}")
    
    # Predict final outcome
    print("\n5. Counterfactual: What if Argentina won?")
    print("-" * 70)
    
    prob_germany = model_full.predict_probability(("Germany", "Argentina"))
    prob_argentina = model_full.predict_probability(("Argentina", "Germany"))
    
    print(f"  P(Germany beats Argentina) = {prob_germany:.4f}")
    print(f"  P(Argentina beats Germany) = {prob_argentina:.4f}")
    print(f"  Germany was favored by {(prob_germany/prob_argentina):.2f}x")

if __name__ == "__main__":
    main()
