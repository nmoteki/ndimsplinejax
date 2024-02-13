import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import ArrayLike


class SplineInterpolant:
    """Spline Interpolant.

    Auto-differencible and Jittable N-dimensitonal spline interpolant using Google/JAX
    Current code supports only 3 and 4 dimensions (N=3 or 4), which are used for CAS data analysis

    Author:
        N.Moteki, (The University of Tokyo, NOAA Earth System Research Lab).

    Assumptions:
        x space (independent valiables) is N-dimension
        Equidistant x-grid in each dimension
        y datum (single real value) is given at each grid point

    User's Inputs:
        a: N-list of lower boundary of x-space [1st-dim, 2nd-dim,...].
        b: N-list of upper boundary of x-space [1st-dim, 2nd-dim,...].
        n: N-list of the number of grid intervals in each dimension.
        c: N-dimensional numpy array (dtype=float) of spline coeffcieints computed using "SplineCoefs_from_GriddedData" module

    Output:
        s3D(x): Autodifferencible and jittable spline interpolant for 3-dim x input vector
        s4D(x): Autodifferencible and jittable spline interpolant for 4-dim x input vector.

    Usage:
        from SplineInterpolant import SplineInterpolant  # import this module
        spline= SplineInterpolant(a,b,n,c_i1...iN) # constructor
        y= spline.sND(x) # evaluate the interpolated y value at the input x vector, where the sND is s3D (if N=3) or s4D (if N=4).
        spline.sND is a jittable and auto-differentiable function with respect to x

    Ref.
    Habermann, C., & Kindermann, F. (2007). Multidimensional spline interpolation: Theory and applications. Computational Economics, 30(2), 153-169.
    Notation is modified by N.Moteki as Note of 2022 September 23-27th

    Created on Fri Oct 21 13:41:11 2022

    @author: moteki
    """

    def __init__(self, a: ArrayLike, b: ArrayLike, n: ArrayLike, c: ArrayLike) -> None:

        self.N = len(a)  # dimension of the problem
        self.a = jnp.array(
            a
        )  # list of lower bound of x-coordinate in each dimension # [1st dim, 2nd dim, ... ]
        self.b = jnp.array(
            b
        )  # list of uppder bound of x-coordinate in each dimension # [1st dim, 2nd dim, ... ]
        self.n = jnp.array(n)  # number of grid interval n in each dimension
        self.h = (self.b - self.a) / self.n  # grid interval in each dimension
        self.c = jnp.array(
            c
        )  # N-dimensional numpy array of y-data ydata[idx1,idx2,...] where the idx1 is the index of grid point along 1st dimension and so forth

    def s1D(self, x: jax.Array) -> jax.Array:
        """1D-spline interpolation.

        INPUTs
        x: 1-dim x vector (float) at which interplated y-value is evaluated
        a: 1-dim vector (float) of the lower boundary of the each of the x-dimension
        h: 1-dim vector (float) of the grid interval of the each of the x-dimension
        c: spline coefficent (1-dim array)
        """

        def u(ii, aa, hh, xx):
            t = jnp.abs((xx - aa) / hh + 2 - ii)
            return lax.cond(
                t <= 1,
                lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
                lambda t: (2.0 - t) ** 3,
                t,
            ) * jnp.heaviside(2.0 - t, 1.0)

        def f(carry, i1, x):
            val = self.c[i1 - 1] * u(i1, self.a[0], self.h[0], x[0])
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)

        carry, val = lax.scan(lambda s1, i1: f(s1, i1=i1, x=x), 0.0, i1arr)

        return carry

    def s2D(self, x: jax.Array) -> jax.Array:
        """2D-spline interpolation.

        INPUTs
        x: 2-dim x vector (float) at which interplated y-value is evaluated
        a: 2-dim vector (float) of the lower boundary of the each of the x-dimension
        h: 2-dim vector (float) of the grid interval of the each of the x-dimension
        c: spline coefficent (2-dim array)
        """

        def u(ii, aa, hh, xx):
            t = jnp.abs((xx - aa) / hh + 2 - ii)
            return lax.cond(
                t <= 1,
                lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
                lambda t: (2.0 - t) ** 3,
                t,
            ) * jnp.heaviside(2.0 - t, 1.0)

        def f(carry, i1, i2, x):
            val = (
                self.c[i1 - 1, i2 - 1]
                * u(i1, self.a[0], self.h[0], x[0])
                * u(i2, self.a[1], self.h[1], x[1])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(lambda s2, i2: f(s2, i1=i1, i2=i2, x=x), s1, i2arr),
            0.0,
            i1arr,
        )

        return carry

    def s3D(self, x: jax.Array) -> jax.Array:
        """3D-spline interpolation.

        INPUTs
        x: 3-dim x vector (float) at which interplated y-value is evaluated
        a: 3-dim vector (float) of the lower boundary of the each of the x-dimension
        h: 3-dim vector (float) of the grid interval of the each of the x-dimension
        c: spline coefficent (3-dim array)
        """

        def u(ii, aa, hh, xx):
            t = jnp.abs((xx - aa) / hh + 2 - ii)
            return lax.cond(
                t <= 1,
                lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
                lambda t: (2.0 - t) ** 3,
                t,
            ) * jnp.heaviside(2.0 - t, 1.0)

        def f(carry, i1, i2, i3, x):
            val = (
                self.c[i1 - 1, i2 - 1, i3 - 1]
                * u(i1, self.a[0], self.h[0], x[0])
                * u(i2, self.a[1], self.h[1], x[1])
                * u(i3, self.a[2], self.h[2], x[2])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)
        i3arr = jnp.arange(1, self.c.shape[2] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(
                lambda s2, i2: lax.scan(
                    lambda s3, i3: f(s3, i1=i1, i2=i2, i3=i3, x=x), s2, i3arr
                ),
                s1,
                i2arr,
            ),
            0.0,
            i1arr,
        )

        return carry

    def s4D(self, x: jax.Array) -> jax.Array:
        """4D-spline interpolation.

        INPUTs
        x: 4-dim x vector (float) at which interplated y-value is evaluated
        a: 4-dim vector (float) of the lower boundary of the each of the x-dimension
        h: 4-dim vector (float) of the grid interval of the each of the x-dimension
        c: spline coefficent (4-dim array)
        """

        def u(ii, aa, hh, xx):
            t = jnp.abs((xx - aa) / hh + 2 - ii)
            return lax.cond(
                t <= 1,
                lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
                lambda t: (2.0 - t) ** 3,
                t,
            ) * jnp.heaviside(2.0 - t, 1.0)

        def f(carry, i1, i2, i3, i4, x):
            val = (
                self.c[i1 - 1, i2 - 1, i3 - 1, i4 - 1]
                * u(i1, self.a[0], self.h[0], x[0])
                * u(i2, self.a[1], self.h[1], x[1])
                * u(i3, self.a[2], self.h[2], x[2])
                * u(i4, self.a[3], self.h[3], x[3])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)
        i3arr = jnp.arange(1, self.c.shape[2] + 1)
        i4arr = jnp.arange(1, self.c.shape[3] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(
                lambda s2, i2: lax.scan(
                    lambda s3, i3: lax.scan(
                        lambda s4, i4: f(s4, i1=i1, i2=i2, i3=i3, i4=i4, x=x), s3, i4arr
                    ),
                    s2,
                    i3arr,
                ),
                s1,
                i2arr,
            ),
            0.0,
            i1arr,
        )

        return carry

    def s5D(self, x: jax.Array) -> jax.Array:
        """5D-spline interpolation.

        INPUTs
        x: 5-dim x vector (float) at which interplated y-value is evaluated
        a: 5-dim vector (float) of the lower boundary of the each of the x-dimension
        h: 5-dim vector (float) of the grid interval of the each of the x-dimension
        c: spline coefficent (5-dim array)
        """

        def u(ii, aa, hh, xx):
            t = jnp.abs((xx - aa) / hh + 2 - ii)
            return lax.cond(
                t <= 1,
                lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3,
                lambda t: (2.0 - t) ** 3,
                t,
            ) * jnp.heaviside(2.0 - t, 1.0)

        def f(carry, i1, i2, i3, i4, i5, x):
            val = (
                self.c[i1 - 1, i2 - 1, i3 - 1, i4 - 1, i5 - 1]
                * u(i1, self.a[0], self.h[0], x[0])
                * u(i2, self.a[1], self.h[1], x[1])
                * u(i3, self.a[2], self.h[2], x[2])
                * u(i4, self.a[3], self.h[3], x[3])
                * u(i5, self.a[4], self.h[4], x[4])
            )
            carry += val
            return carry, val

        i1arr = jnp.arange(1, self.c.shape[0] + 1)
        i2arr = jnp.arange(1, self.c.shape[1] + 1)
        i3arr = jnp.arange(1, self.c.shape[2] + 1)
        i4arr = jnp.arange(1, self.c.shape[3] + 1)
        i5arr = jnp.arange(1, self.c.shape[4] + 1)

        carry, val = lax.scan(
            lambda s1, i1: lax.scan(
                lambda s2, i2: lax.scan(
                    lambda s3, i3: lax.scan(
                        lambda s4, i4: lax.scan(
                            lambda s5, i5: f(
                                s5, i1=i1, i2=i2, i3=i3, i4=i4, i5=i5, x=x
                            ),
                            s4,
                            i5arr,
                        ),
                        s3,
                        i4arr,
                    ),
                    s2,
                    i3arr,
                ),
                s1,
                i2arr,
            ),
            0.0,
            i1arr,
        )

        return carry
