"""Test 5-dimensional grid.

A example script for illustaring the usage of "SplineCoefs_from_GriddedData" and
"SplineInterpolant modules" to obtain jittable and auto-differentible
multidimentional spline interpolant.

Created on Fri Oct 21 12:29:47 2022

@author: moteki
"""

import jax.numpy as jnp
from jax import grad, jit

from ndimsplinejax import SplineCoefs_from_GriddedData, SplineInterpolant


def test_5d() -> None:
    """Test 5-dimensional grid."""
    #### synthetic data for demonstration (5-dimension) ####
    a = [0, 0, 0, 0, 0]
    b = [1, 2, 3, 4, 5]
    n = [10, 10, 10, 10, 10]
    N = len(a)

    x_grid = tuple(jnp.linspace(a[j], b[j], n[j] + 1) for j in range(N))
    grid_shape = tuple(n[j] + 1 for j in range(N))

    # Assuming x_grid is a list of 1D arrays
    grids = jnp.meshgrid(*x_grid, indexing="ij")

    # Apply the sin function to each grid and reduce by multiplication
    y_data = jnp.prod(jnp.asarray([jnp.sin(grid) for grid in grids]), axis=0)

    # compute spline coefficients from the gridded data
    spline_coef = SplineCoefs_from_GriddedData(a, b, y_data)
    c_i1i2i3i4i5 = spline_coef.compute_coeffs()

    # compute the jittable and auto-differentiable spline interpolant using the
    # coefficient.
    spline = SplineInterpolant(a, b, n, c_i1i2i3i4i5)

    # give a particular x-coordinate for function evaluation
    x = jnp.array([0.7, 1.0, 1.5, 2.0, 2.5])

    y = spline.s5D(x)

    s5d_jitted = jit(spline.s5D)
    assert jnp.allclose(s5d_jitted(x), y)

    ds5d = grad(s5d_jitted)
    grady = ds5d(x)
    assert jnp.isfinite(grady).all()
