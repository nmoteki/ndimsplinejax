"""N-dimensional splines in JAX."""

__all__ = ["SplineCoefs_from_GriddedData", "SplineInterpolant"]

from .coeffs import SplineCoefs_from_GriddedData
from .core import SplineInterpolant
