from .least_squares import generic_fit as least_squares_fit
from .orthogonal_distance import generic_fit as orthogonal_distance_regression_fit

from .fitResult import FitResult
from .fitModel import FitModel
from .models import Linear, InverseSquare, Exponential, LimitedGrowth, ResonanceCurve, AmpTiefpass

__all__ = [
    "least_squares_fit",
    "orthogonal_distance_regression_fit",
    "FitResult",
    "FitModel",
    "Linear",
    "InverseSquare",
    "Exponential",
    "LimitedGrowth",
    "ResonanceCurve",
    "AmpTiefpass"
]

