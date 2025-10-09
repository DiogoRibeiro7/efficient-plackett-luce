"""
Projected pairwise Plackett-Luce models.

Converts multi-body comparisons to pairwise comparisons for comparison
with the full multi-body model.
"""

from .core import PlackettLuceModel
from typing import List, Tuple


class ProjectedPlackettLuce(PlackettLuceModel):
    """
    Projected Plackett-Luce model that converts multi-body comparisons
    to pairwise comparisons (Eq. 16 and 17 from paper).
    """
    
    def _project_to_pairwise(self, hyperedges: List[Tuple]) -> List[Tuple]:
        """
        Project multi-body comparisons to pairwise comparisons.
        
        For full model: Creates all pairwise comparisons preserving order
        For position1: Creates pairwise comparisons only for first vs rest
        """
        pairwise_edges = []
        
        for ranking, weight in hyperedges:
            if self.model_type == 'full':
                # Full breaking: all pairs (i,j) where i ranks before j
                for i in range(len(ranking)):
                    for j in range(i + 1, len(ranking)):
                        pairwise_edges.append(((ranking[i], ranking[j]), weight))
            else:  # position1
                # Position-1 breaking: first vs all others
                for j in range(1, len(ranking)):
                    pairwise_edges.append(((ranking[0], ranking[j]), weight))
        
        return pairwise_edges
    
    def fit(self, hyperedges: List[Tuple], verbose: bool = False) -> Dict:
        """Fit model on projected pairwise comparisons."""
        pairwise = self._project_to_pairwise(hyperedges)
        return super().fit(pairwise, verbose)
