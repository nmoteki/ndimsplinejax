"""N-dimensional splines in JAX."""

__all__ = ["SplineCoefs_from_GriddedData", "SplineInterpolant"]

import jaxtyping

with jaxtyping.install_import_hook("ndimsplinejax", "beartype.beartype"):
    from .coeffs import SplineCoefs_from_GriddedData
    from .core import SplineInterpolant


# Clean up namespace
del jaxtyping
