"""
Efficient Plackett-Luce: Fast inference for multi-body rankings.

This package implements Newman's efficient algorithm for the Plackett-Luce
model, achieving 5-70x speedup over traditional methods.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .core import PlackettLuceModel
from .projected import ProjectedPlackettLuce
from .validation import cross_validate, train_test_split
from .utils import generate_synthetic_rankings, ranking_similarity

__all__ = [
    "PlackettLuceModel",
    "ProjectedPlackettLuce",
    "cross_validate",
    "train_test_split",
    "generate_synthetic_rankings",
    "ranking_similarity",
]
