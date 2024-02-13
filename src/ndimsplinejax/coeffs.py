"""Module for computing the spline coefficients from gridded data.

Compute the coefficients of the N-dimensitonal natural-cubic spline interpolant
defined by Habermann and Kindermann 2007.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import NDArray
from scipy import linalg


class SplineCoefs_from_GriddedData(eqx.Module):  # type: ignore[misc]
    """Compute the coeffcieits.

    Compute the coefficients of the N-dimensitonal natural-cubic spline
    interpolant defined by Habermann and Kindermann 2007 Current code supports
    up to 5 dimensions (N can be either of 1,2,3,4,5).

    Author:
        N.Moteki, (The University of Tokyo, NOAA Earth System Research Lab).

    Assumptions:
        x space (independent variables) is N-dimension Equidistant x-grid in
        each dimension y datum (single real value) is given at each grid point

    User's Inputs:
        a: N-list of lower boundary of x-space [1st-dim, 2nd-dim,...].  b:
        N-list of upper boundary of x-space [1st-dim, 2nd-dim,...].  y_data:
        N-dimensional numpy array of data (the value of dependent variable y) on
        the x-gridpoints.

    Output:
        c_i1...iN: N-dimensional numpy array (dtype=float) of spline
        coeffcieints defined as HK2007 p161.

    Usage:
        >>> from ``SplineCoefs_from_GriddedData`` import SplineCoefs_from_GriddedData
        >>> spline_coef = SplineCoefs_from_GriddedData(a,b,n,y_data)
        >>> spline_coef.compute_coeffs()


    Ref.  Habermann, C., & Kindermann, F. (2007). Multidimensional spline
    interpolation: Theory and applications. Computational Economics, 30(2),
    153-169.  Notation is modified by N.Moteki as Note of 2022 September 23-27th

    Created on Fri Oct 21 13:41:11 2022
    """

    a: Float[np.ndarray, "N"] = eqx.field(
        converter=lambda x: np.array(x, dtype=float), static=True
    )
    b: Float[np.ndarray, "N"] = eqx.field(
        converter=lambda x: np.array(x, dtype=float), static=True
    )
    y_data: Float[np.ndarray, "..."] = eqx.field(
        converter=lambda x: np.array(x, dtype=float), static=True
    )

    @property
    def N(self) -> int:
        """Dimension of the problem."""
        return len(self.a)

    @property
    def n(self) -> NDArray[np.integer]:
        """Number of grid interval n in each dimension."""
        return np.asarray(self.y_data.shape, dtype=int) - 1

    def get_c_shape(self, k: int) -> tuple[int, ...]:
        """Get the shape of the coefficient array."""
        n = self.n
        return tuple(int(n[j]) + (3 if j <= k else 1) for j in range(self.N))

    def _compute_coefs_1d(self) -> Float[Array, "..."]:
        n = self.n
        k = 0  # 1-st dimension
        c_i1 = np.zeros(self.get_c_shape(k))
        c_i1[1] = self.y_data[0] / 6  # c_{2}
        c_i1[n[k] + 1] = self.y_data[n[k]] / 6  # c_{n+2}
        A = np.eye(n[k] - 1) * 4 + np.eye(n[k] - 1, k=1) + np.eye(n[k] - 1, k=-1)
        B = np.zeros(n[k] - 1)
        B[0] = self.y_data[1] - c_i1[1]
        B[n[k] - 2] = self.y_data[n[k] - 1] - c_i1[n[k] + 1]
        B[1 : n[k] - 2] = self.y_data[2 : n[k] - 1]
        sol = linalg.solve(A, B)
        c_i1[2 : n[k] + 1] = sol
        c_i1[0] = 2 * c_i1[1] - c_i1[2]
        c_i1[n[k] + 2] = 2 * c_i1[n[k] + 1] - c_i1[n[k]]

        return jnp.asarray(c_i1)

    def _compute_coefs_2d(self) -> Float[Array, "..."]:
        n = self.n
        k = 0  # 1-st dimension
        c_i1q2 = np.zeros(self.get_c_shape(k))
        for q2 in range(n[1] + 1):
            c_i1q2[1, q2] = self.y_data[0, q2] / 6  # c_{2}
            c_i1q2[n[k] + 1, q2] = self.y_data[n[k], q2] / 6  # c_{n+2}
            A = np.eye(n[k] - 1) * 4 + np.eye(n[k] - 1, k=1) + np.eye(n[k] - 1, k=-1)
            B = np.zeros(n[k] - 1)
            B[0] = self.y_data[1, q2] - c_i1q2[1, q2]
            B[n[k] - 2] = self.y_data[n[k] - 1, q2] - c_i1q2[n[k] + 1, q2]
            B[1 : n[k] - 2] = self.y_data[2 : n[k] - 1, q2]
            sol = linalg.solve(A, B)
            c_i1q2[2 : n[k] + 1, q2] = sol
            c_i1q2[0, q2] = 2 * c_i1q2[1, q2] - c_i1q2[2, q2]
            c_i1q2[n[k] + 2, q2] = 2 * c_i1q2[n[k] + 1, q2] - c_i1q2[n[k], q2]

        k = 1  # 2nd dimension
        c_i1i2 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            c_i1i2[i1, 1] = c_i1q2[i1, 0] / 6  # c_{2}
            c_i1i2[i1, n[k] + 1] = c_i1q2[i1, n[k]] / 6  # c_{n+2}
            A = np.eye(n[k] - 1) * 4 + np.eye(n[k] - 1, k=1) + np.eye(n[k] - 1, k=-1)
            B = np.zeros(n[k] - 1)
            B[0] = c_i1q2[i1, 1] - c_i1i2[i1, 1]
            B[n[k] - 2] = c_i1q2[i1, n[k] - 1] - c_i1i2[i1, n[k] + 1]
            B[1 : n[k] - 2] = c_i1q2[i1, 2 : n[k] - 1]
            sol = linalg.solve(A, B)
            c_i1i2[i1, 2 : n[k] + 1] = sol
            c_i1i2[i1, 0] = 2 * c_i1i2[i1, 1] - c_i1i2[i1, 2]
            c_i1i2[i1, n[k] + 2] = 2 * c_i1i2[i1, n[k] + 1] - c_i1i2[i1, n[k]]

        return jnp.asarray(c_i1i2)

    def _compute_coefs_3d(self) -> Float[Array, "..."]:
        n = self.n
        k = 0  # 1-st dimension
        c_i1q2q3 = np.zeros(self.get_c_shape(k))
        for q2 in range(n[1] + 1):
            for q3 in range(n[2] + 1):
                c_i1q2q3[1, q2, q3] = self.y_data[0, q2, q3] / 6  # c_{2}
                c_i1q2q3[n[k] + 1, q2, q3] = self.y_data[n[k], q2, q3] / 6  # c_{n+2}
                A = (
                    np.eye(n[k] - 1) * 4
                    + np.eye(n[k] - 1, k=1)
                    + np.eye(n[k] - 1, k=-1)
                )
                B = np.zeros(n[k] - 1)
                B[0] = self.y_data[1, q2, q3] - c_i1q2q3[1, q2, q3]
                B[n[k] - 2] = self.y_data[n[k] - 1, q2, q3] - c_i1q2q3[n[k] + 1, q2, q3]
                B[1 : n[k] - 2] = self.y_data[2 : n[k] - 1, q2, q3]
                sol = linalg.solve(A, B)
                c_i1q2q3[2 : n[k] + 1, q2, q3] = sol
                c_i1q2q3[0, q2, q3] = 2 * c_i1q2q3[1, q2, q3] - c_i1q2q3[2, q2, q3]
                c_i1q2q3[n[k] + 2, q2, q3] = (
                    2 * c_i1q2q3[n[k] + 1, q2, q3] - c_i1q2q3[n[k], q2, q3]
                )

        k = 1  # 2nd dimension
        c_i1i2q3 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for q3 in range(n[2] + 1):
                c_i1i2q3[i1, 1, q3] = c_i1q2q3[i1, 0, q3] / 6  # c_{2}
                c_i1i2q3[i1, n[k] + 1, q3] = c_i1q2q3[i1, n[k], q3] / 6  # c_{n+2}
                A = (
                    np.eye(n[k] - 1) * 4
                    + np.eye(n[k] - 1, k=1)
                    + np.eye(n[k] - 1, k=-1)
                )
                B = np.zeros(n[k] - 1)
                B[0] = c_i1q2q3[i1, 1, q3] - c_i1i2q3[i1, 1, q3]
                B[n[k] - 2] = c_i1q2q3[i1, n[k] - 1, q3] - c_i1i2q3[i1, n[k] + 1, q3]
                B[1 : n[k] - 2] = c_i1q2q3[i1, 2 : n[k] - 1, q3]
                sol = linalg.solve(A, B)
                c_i1i2q3[i1, 2 : n[k] + 1, q3] = sol
                c_i1i2q3[i1, 0, q3] = 2 * c_i1i2q3[i1, 1, q3] - c_i1i2q3[i1, 2, q3]
                c_i1i2q3[i1, n[k] + 2, q3] = (
                    2 * c_i1i2q3[i1, n[k] + 1, q3] - c_i1i2q3[i1, n[k], q3]
                )

        k = 2  # 3rd dimension
        c_i1i2i3 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for i2 in range(n[1] + 3):
                c_i1i2i3[i1, i2, 1] = c_i1i2q3[i1, i2, 0] / 6  # c_{2}
                c_i1i2i3[i1, i2, n[k] + 1] = c_i1i2q3[i1, i2, n[k]] / 6  # c_{n+2}
                A = (
                    np.eye(n[k] - 1) * 4
                    + np.eye(n[k] - 1, k=1)
                    + np.eye(n[k] - 1, k=-1)
                )
                B = np.zeros(n[k] - 1)
                B[0] = c_i1i2q3[i1, i2, 1] - c_i1i2i3[i1, i2, 1]
                B[n[k] - 2] = c_i1i2q3[i1, i2, n[k] - 1] - c_i1i2i3[i1, i2, n[k] + 1]
                B[1 : n[k] - 2] = c_i1i2q3[i1, i2, 2 : n[k] - 1]
                sol = linalg.solve(A, B)
                c_i1i2i3[i1, i2, 2 : n[k] + 1] = sol
                c_i1i2i3[i1, i2, 0] = 2 * c_i1i2i3[i1, i2, 1] - c_i1i2i3[i1, i2, 2]
                c_i1i2i3[i1, i2, n[k] + 2] = (
                    2 * c_i1i2i3[i1, i2, n[k] + 1] - c_i1i2i3[i1, i2, n[k]]
                )

        return jnp.asarray(c_i1i2i3)

    def _compute_coefs_4d(self) -> Float[Array, "..."]:  # noqa: C901
        n = self.n
        k = 0  # 1st dimension
        c_i1q2q3q4 = np.zeros(self.get_c_shape(k))
        for q2 in range(n[1] + 1):
            for q3 in range(n[2] + 1):
                for q4 in range(n[3] + 1):
                    c_i1q2q3q4[1, q2, q3, q4] = self.y_data[0, q2, q3, q4] / 6  # c_{2}
                    c_i1q2q3q4[n[k] + 1, q2, q3, q4] = (
                        self.y_data[n[k], q2, q3, q4] / 6
                    )  # c_{n+2}
                    A = (
                        np.eye(n[k] - 1) * 4
                        + np.eye(n[k] - 1, k=1)
                        + np.eye(n[k] - 1, k=-1)
                    )
                    B = np.zeros(n[k] - 1)
                    B[0] = self.y_data[1, q2, q3, q4] - c_i1q2q3q4[1, q2, q3, q4]
                    B[n[k] - 2] = (
                        self.y_data[n[k] - 1, q2, q3, q4]
                        - c_i1q2q3q4[n[k] + 1, q2, q3, q4]
                    )
                    B[1 : n[k] - 2] = self.y_data[2 : n[k] - 1, q2, q3, q4]
                    sol = linalg.solve(A, B)
                    c_i1q2q3q4[2 : n[k] + 1, q2, q3, q4] = sol
                    c_i1q2q3q4[0, q2, q3, q4] = (
                        2 * c_i1q2q3q4[1, q2, q3, q4] - c_i1q2q3q4[2, q2, q3, q4]
                    )
                    c_i1q2q3q4[n[k] + 2, q2, q3, q4] = (
                        2 * c_i1q2q3q4[n[k] + 1, q2, q3, q4]
                        - c_i1q2q3q4[n[k], q2, q3, q4]
                    )

        k = 1  # 2nd dimension
        c_i1i2q3q4 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for q3 in range(n[2] + 1):
                for q4 in range(n[3] + 1):
                    c_i1i2q3q4[i1, 1, q3, q4] = c_i1q2q3q4[i1, 0, q3, q4] / 6  # c_{2}
                    c_i1i2q3q4[i1, n[k] + 1, q3, q4] = (
                        c_i1q2q3q4[i1, n[k], q3, q4] / 6
                    )  # c_{n+2}
                    A = (
                        np.eye(n[k] - 1) * 4
                        + np.eye(n[k] - 1, k=1)
                        + np.eye(n[k] - 1, k=-1)
                    )
                    B = np.zeros(n[k] - 1)
                    B[0] = c_i1q2q3q4[i1, 1, q3, q4] - c_i1i2q3q4[i1, 1, q3, q4]
                    B[n[k] - 2] = (
                        c_i1q2q3q4[i1, n[k] - 1, q3, q4]
                        - c_i1i2q3q4[i1, n[k] + 1, q3, q4]
                    )
                    B[1 : n[k] - 2] = c_i1q2q3q4[i1, 2 : n[k] - 1, q3, q4]
                    sol = linalg.solve(A, B)
                    c_i1i2q3q4[i1, 2 : n[k] + 1, q3, q4] = sol
                    c_i1i2q3q4[i1, 0, q3, q4] = (
                        2 * c_i1i2q3q4[i1, 1, q3, q4] - c_i1i2q3q4[i1, 2, q3, q4]
                    )
                    c_i1i2q3q4[i1, n[k] + 2, q3, q4] = (
                        2 * c_i1i2q3q4[i1, n[k] + 1, q3, q4]
                        - c_i1i2q3q4[i1, n[k], q3, q4]
                    )

        k = 2  # 3rd dimension
        c_i1i2i3q4 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for i2 in range(n[1] + 3):
                for q4 in range(n[3] + 1):
                    c_i1i2i3q4[i1, i2, 1, q4] = c_i1i2q3q4[i1, i2, 0, q4] / 6  # c_{2}
                    c_i1i2i3q4[i1, i2, n[k] + 1, q4] = (
                        c_i1i2q3q4[i1, i2, n[k], q4] / 6
                    )  # c_{n+2}
                    A = (
                        np.eye(n[k] - 1) * 4
                        + np.eye(n[k] - 1, k=1)
                        + np.eye(n[k] - 1, k=-1)
                    )
                    B = np.zeros(n[k] - 1)
                    B[0] = c_i1i2q3q4[i1, i2, 1, q4] - c_i1i2i3q4[i1, i2, 1, q4]
                    B[n[k] - 2] = (
                        c_i1i2q3q4[i1, i2, n[k] - 1, q4]
                        - c_i1i2i3q4[i1, i2, n[k] + 1, q4]
                    )
                    B[1 : n[k] - 2] = c_i1i2q3q4[i1, i2, 2 : n[k] - 1, q4]
                    sol = linalg.solve(A, B)
                    c_i1i2i3q4[i1, i2, 2 : n[k] + 1, q4] = sol
                    c_i1i2i3q4[i1, i2, 0, q4] = (
                        2 * c_i1i2i3q4[i1, i2, 1, q4] - c_i1i2i3q4[i1, i2, 2, q4]
                    )
                    c_i1i2i3q4[i1, i2, n[k] + 2, q4] = (
                        2 * c_i1i2i3q4[i1, i2, n[k] + 1, q4]
                        - c_i1i2i3q4[i1, i2, n[k], q4]
                    )

        k = 3  # 4th dimension
        c_i1i2i3i4 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for i2 in range(n[1] + 3):
                for i3 in range(n[2] + 3):
                    c_i1i2i3i4[i1, i2, i3, 1] = c_i1i2i3q4[i1, i2, i3, 0] / 6  # c_{2}
                    c_i1i2i3i4[i1, i2, i3, n[k] + 1] = (
                        c_i1i2i3q4[i1, i2, i3, n[k]] / 6
                    )  # c_{n+2}
                    A = (
                        np.eye(n[k] - 1) * 4
                        + np.eye(n[k] - 1, k=1)
                        + np.eye(n[k] - 1, k=-1)
                    )
                    B = np.zeros(n[k] - 1)
                    B[0] = c_i1i2i3q4[i1, i2, i3, 1] - c_i1i2i3i4[i1, i2, i3, 1]
                    B[n[k] - 2] = (
                        c_i1i2i3q4[i1, i2, i3, n[k] - 1]
                        - c_i1i2i3i4[i1, i2, i3, n[k] + 1]
                    )
                    B[1 : n[k] - 2] = c_i1i2i3q4[i1, i2, i3, 2 : n[k] - 1]
                    sol = linalg.solve(A, B)
                    c_i1i2i3i4[i1, i2, i3, 2 : n[k] + 1] = sol
                    c_i1i2i3i4[i1, i2, i3, 0] = (
                        2 * c_i1i2i3i4[i1, i2, i3, 1] - c_i1i2i3i4[i1, i2, i3, 2]
                    )
                    c_i1i2i3i4[i1, i2, i3, n[k] + 2] = (
                        2 * c_i1i2i3i4[i1, i2, i3, n[k] + 1]
                        - c_i1i2i3i4[i1, i2, i3, n[k]]
                    )

        return jnp.asarray(c_i1i2i3i4)

    def _compute_coefs_5d(self) -> Float[Array, "..."]:  # noqa: C901
        n = self.n
        k = 0  # 1st dimension
        c_i1q2q3q4q5 = np.zeros(self.get_c_shape(k))
        for q2 in range(n[1] + 1):
            for q3 in range(n[2] + 1):
                for q4 in range(n[3] + 1):
                    for q5 in range(n[4] + 1):
                        c_i1q2q3q4q5[1, q2, q3, q4, q5] = (
                            self.y_data[0, q2, q3, q4, q5] / 6
                        )  # c_{2}
                        c_i1q2q3q4q5[n[k] + 1, q2, q3, q4, q5] = (
                            self.y_data[n[k], q2, q3, q4, q5] / 6
                        )  # c_{n+2}
                        A = (
                            np.eye(n[k] - 1) * 4
                            + np.eye(n[k] - 1, k=1)
                            + np.eye(n[k] - 1, k=-1)
                        )
                        B = np.zeros(n[k] - 1)
                        B[0] = (
                            self.y_data[1, q2, q3, q4, q5]
                            - c_i1q2q3q4q5[1, q2, q3, q4, q5]
                        )
                        B[n[k] - 2] = (
                            self.y_data[n[k] - 1, q2, q3, q4, q5]
                            - c_i1q2q3q4q5[n[k] + 1, q2, q3, q4, q5]
                        )
                        B[1 : n[k] - 2] = self.y_data[2 : n[k] - 1, q2, q3, q4, q5]
                        sol = linalg.solve(A, B)
                        c_i1q2q3q4q5[2 : n[k] + 1, q2, q3, q4, q5] = sol
                        c_i1q2q3q4q5[0, q2, q3, q4, q5] = (
                            2 * c_i1q2q3q4q5[1, q2, q3, q4, q5]
                            - c_i1q2q3q4q5[2, q2, q3, q4, q5]
                        )
                        c_i1q2q3q4q5[n[k] + 2, q2, q3, q4, q5] = (
                            2 * c_i1q2q3q4q5[n[k] + 1, q2, q3, q4, q5]
                            - c_i1q2q3q4q5[n[k], q2, q3, q4, q5]
                        )

        k = 1  # 2nd dimension
        c_i1i2q3q4q5 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for q3 in range(n[2] + 1):
                for q4 in range(n[3] + 1):
                    for q5 in range(n[4] + 1):
                        c_i1i2q3q4q5[i1, 1, q3, q4, q5] = (
                            c_i1q2q3q4q5[i1, 0, q3, q4, q5] / 6
                        )  # c_{2}
                        c_i1i2q3q4q5[i1, n[k] + 1, q3, q4, q5] = (
                            c_i1q2q3q4q5[i1, n[k], q3, q4, q5] / 6
                        )  # c_{n+2}
                        A = (
                            np.eye(n[k] - 1) * 4
                            + np.eye(n[k] - 1, k=1)
                            + np.eye(n[k] - 1, k=-1)
                        )
                        B = np.zeros(n[k] - 1)
                        B[0] = (
                            c_i1q2q3q4q5[i1, 1, q3, q4, q5]
                            - c_i1i2q3q4q5[i1, 1, q3, q4, q5]
                        )
                        B[n[k] - 2] = (
                            c_i1q2q3q4q5[i1, n[k] - 1, q3, q4, q5]
                            - c_i1i2q3q4q5[i1, n[k] + 1, q3, q4, q5]
                        )
                        B[1 : n[k] - 2] = c_i1q2q3q4q5[i1, 2 : n[k] - 1, q3, q4, q5]
                        sol = linalg.solve(A, B)
                        c_i1i2q3q4q5[i1, 2 : n[k] + 1, q3, q4, q5] = sol
                        c_i1i2q3q4q5[i1, 0, q3, q4] = (
                            2 * c_i1i2q3q4q5[i1, 1, q3, q4, q5]
                            - c_i1i2q3q4q5[i1, 2, q3, q4, q5]
                        )
                        c_i1i2q3q4q5[i1, n[k] + 2, q3, q4, q5] = (
                            2 * c_i1i2q3q4q5[i1, n[k] + 1, q3, q4, q5]
                            - c_i1i2q3q4q5[i1, n[k], q3, q4, q5]
                        )

        k = 2  # 3rd dimension
        c_i1i2i3q4q5 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for i2 in range(n[1] + 3):
                for q4 in range(n[3] + 1):
                    for q5 in range(n[4] + 1):
                        c_i1i2i3q4q5[i1, i2, 1, q4, q5] = (
                            c_i1i2q3q4q5[i1, i2, 0, q4, q5] / 6
                        )  # c_{2}
                        c_i1i2i3q4q5[i1, i2, n[k] + 1, q4, q5] = (
                            c_i1i2q3q4q5[i1, i2, n[k], q4, q5] / 6
                        )  # c_{n+2}
                        A = (
                            np.eye(n[k] - 1) * 4
                            + np.eye(n[k] - 1, k=1)
                            + np.eye(n[k] - 1, k=-1)
                        )
                        B = np.zeros(n[k] - 1)
                        B[0] = (
                            c_i1i2q3q4q5[i1, i2, 1, q4, q5]
                            - c_i1i2i3q4q5[i1, i2, 1, q4, q5]
                        )
                        B[n[k] - 2] = (
                            c_i1i2q3q4q5[i1, i2, n[k] - 1, q4, q5]
                            - c_i1i2i3q4q5[i1, i2, n[k] + 1, q4, q5]
                        )
                        B[1 : n[k] - 2] = c_i1i2q3q4q5[i1, i2, 2 : n[k] - 1, q4, q5]
                        sol = linalg.solve(A, B)
                        c_i1i2i3q4q5[i1, i2, 2 : n[k] + 1, q4, q5] = sol
                        c_i1i2i3q4q5[i1, i2, 0, q4, q5] = (
                            2 * c_i1i2i3q4q5[i1, i2, 1, q4, q5]
                            - c_i1i2i3q4q5[i1, i2, 2, q4, q5]
                        )
                        c_i1i2i3q4q5[i1, i2, n[k] + 2, q4, q5] = (
                            2 * c_i1i2i3q4q5[i1, i2, n[k] + 1, q4, q5]
                            - c_i1i2i3q4q5[i1, i2, n[k], q4, q5]
                        )

        k = 3  # 4th dimension
        c_i1i2i3i4q5 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for i2 in range(n[1] + 3):
                for i3 in range(n[2] + 3):
                    for q5 in range(n[4] + 1):
                        c_i1i2i3i4q5[i1, i2, i3, 1, q5] = (
                            c_i1i2i3q4q5[i1, i2, i3, 0, q5] / 6
                        )  # c_{2}
                        c_i1i2i3i4q5[i1, i2, i3, n[k] + 1, q5] = (
                            c_i1i2i3q4q5[i1, i2, i3, n[k], q5] / 6
                        )  # c_{n+2}
                        A = (
                            np.eye(n[k] - 1) * 4
                            + np.eye(n[k] - 1, k=1)
                            + np.eye(n[k] - 1, k=-1)
                        )
                        B = np.zeros(n[k] - 1)
                        B[0] = (
                            c_i1i2i3q4q5[i1, i2, i3, 1, q5]
                            - c_i1i2i3i4q5[i1, i2, i3, 1, q5]
                        )
                        B[n[k] - 2] = (
                            c_i1i2i3q4q5[i1, i2, i3, n[k] - 1, q5]
                            - c_i1i2i3i4q5[i1, i2, i3, n[k] + 1, q5]
                        )
                        B[1 : n[k] - 2] = c_i1i2i3q4q5[i1, i2, i3, 2 : n[k] - 1, q5]
                        sol = linalg.solve(A, B)
                        c_i1i2i3i4q5[i1, i2, i3, 2 : n[k] + 1, q5] = sol
                        c_i1i2i3i4q5[i1, i2, i3, 0, q5] = (
                            2 * c_i1i2i3i4q5[i1, i2, i3, 1, q5]
                            - c_i1i2i3i4q5[i1, i2, i3, 2, q5]
                        )
                        c_i1i2i3i4q5[i1, i2, i3, n[k] + 2, q5] = (
                            2 * c_i1i2i3i4q5[i1, i2, i3, n[k] + 1, q5]
                            - c_i1i2i3i4q5[i1, i2, i3, n[k], q5]
                        )

        k = 4  # 5th dimension
        c_i1i2i3i4i5 = np.zeros(self.get_c_shape(k))
        for i1 in range(n[0] + 3):
            for i2 in range(n[1] + 3):
                for i3 in range(n[2] + 3):
                    for i4 in range(n[3] + 3):
                        c_i1i2i3i4i5[i1, i2, i3, i4, 1] = (
                            c_i1i2i3i4q5[i1, i2, i3, i4, 0] / 6
                        )  # c_{2}
                        c_i1i2i3i4i5[i1, i2, i3, i4, n[k] + 1] = (
                            c_i1i2i3i4q5[i1, i2, i3, i4, n[k]] / 6
                        )  # c_{n+2}
                        A = (
                            np.eye(n[k] - 1) * 4
                            + np.eye(n[k] - 1, k=1)
                            + np.eye(n[k] - 1, k=-1)
                        )
                        B = np.zeros(n[k] - 1)
                        B[0] = (
                            c_i1i2i3i4q5[i1, i2, i3, i4, 1]
                            - c_i1i2i3i4i5[i1, i2, i3, i4, 1]
                        )
                        B[n[k] - 2] = (
                            c_i1i2i3i4q5[i1, i2, i3, i4, n[k] - 1]
                            - c_i1i2i3i4i5[i1, i2, i3, i4, n[k] + 1]
                        )
                        B[1 : n[k] - 2] = c_i1i2i3i4q5[i1, i2, i3, i4, 2 : n[k] - 1]
                        sol = linalg.solve(A, B)
                        c_i1i2i3i4i5[i1, i2, i3, i4, 2 : n[k] + 1] = sol
                        c_i1i2i3i4i5[i1, i2, i3, i4, 0] = (
                            2 * c_i1i2i3i4i5[i1, i2, i3, i4, 1]
                            - c_i1i2i3i4i5[i1, i2, i3, i4, 2]
                        )
                        c_i1i2i3i4i5[i1, i2, i3, i4, n[k] + 2] = (
                            2 * c_i1i2i3i4i5[i1, i2, i3, i4, n[k] + 1]
                            - c_i1i2i3i4i5[i1, i2, i3, i4, n[k]]
                        )

        return jnp.asarray(c_i1i2i3i4i5)

    def compute_coeffs(self) -> Float[Array, "..."]:
        """Compute the coefficients for the spline interpolation."""
        N = eqx.error_if(self.N, self.N > 5, "N>=6 is unsupported!")

        if N == 1:
            out = self._compute_coefs_1d()

        elif N == 2:
            out = self._compute_coefs_2d()

        elif N == 3:
            out = self._compute_coefs_3d()

        elif N == 4:
            out = self._compute_coefs_4d()

        elif N == 5:
            out = self._compute_coefs_5d()

        return out
