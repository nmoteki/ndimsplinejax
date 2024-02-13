"""Test 5-dimensional grid.

A example script for illustaring the usage of "SplineCoefs_from_GriddedData" and
"SplineInterpolant modules" to obtain jittable and auto-differentible
multidimentional spline interpolant.

Created on Fri Oct 21 12:29:47 2022

@author: moteki
"""

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, value_and_grad

from ndimsplinejax import SplineCoefs_from_GriddedData, SplineInterpolant

#### synthetic data for demonstration (5-dimension) ####
a = [0, 0, 0, 0, 0]
b = [1, 2, 3, 4, 5]
n = [10, 10, 10, 10, 10]
N = len(a)

x_grid = ()
for j in range(N):
    x_grid += (np.linspace(a[j], b[j], n[j] + 1),)

grid_shape = ()
for j in range(N):
    grid_shape += (n[j] + 1,)
y_data = np.zeros(grid_shape)


for q1 in range(n[0] + 1):
    for q2 in range(n[1] + 1):
        for q3 in range(n[2] + 1):
            for q4 in range(n[3] + 1):
                for q5 in range(n[4] + 1):
                    y_data[q1, q2, q3, q4, q5] = (
                        np.sin(x_grid[0][q1])
                        * np.sin(x_grid[1][q2])
                        * np.sin(x_grid[2][q3])
                        * np.sin(x_grid[3][q4])
                        * np.sin(x_grid[4][q5])
                    )


# compute spline coefficients from the gridded data
spline_coef = SplineCoefs_from_GriddedData(a, b, y_data)
c_i1i2i3i4i5 = spline_coef.Compute_Coefs()

# compute the jittable and auto-differentiable spline interpolant using the coefficient.
spline = SplineInterpolant(a, b, n, c_i1i2i3i4i5)


# give a particular x-coordinate for function evaluation
x = jnp.array([0.7, 1.0, 1.5, 2.0, 2.5])
print(spline.s5D(x))


s5d_jitted = jit(spline.s5D)
print(s5d_jitted(x))


ds5d = grad(spline.s5D)
print(ds5d(x))
ds5d_jitted = jit(grad(spline.s5D))
print(ds5d_jitted(x))

s5d_fun = value_and_grad(spline.s5D)
print(s5d_fun(x))
s5d_fun_jitted = jit(value_and_grad(spline.s5D))
print(s5d_fun_jitted(x))

# %timeit spline.s5D(x)
# %timeit s5d_jitted(x)
# %timeit ds5d(x)
# %timeit ds5d_jitted(x)
# %timeit s5d_fun(x)
# %timeit s5d_fun_jitted(x)
