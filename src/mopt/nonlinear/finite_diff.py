"""Numerical differentiation by central finite differences."""

from __future__ import annotations

from typing import Callable

import numpy as np


def finite_difference_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float | None = None,
) -> np.ndarray:
    """Approximate the gradient of ``f`` at ``x`` by central differences.

    Each component is estimated as

    .. math::

        \\partial_i f(x) \\approx
        \\frac{f(x + h_i e_i) - f(x - h_i e_i)}{2 h_i},

    with a per-coordinate step ``h_i = h * (1 + |x_i|)`` so the step scales
    with the magnitude of ``x``. The truncation error is O(h^2), at the cost
    of ``2 n`` evaluations of ``f``.

    Parameters
    ----------
    f : callable
        Scalar-valued function of a 1-D array.
    x : np.ndarray, shape (n,)
        Point at which to differentiate.
    h : float, optional
        Base step size. Defaults to ``eps ** (1/3)`` (about 6e-6), the
        rule-of-thumb optimum balancing truncation and rounding error for
        central differences.

    Returns
    -------
    np.ndarray, shape (n,)
        Gradient estimate.
    """
    x = np.asarray(x, dtype=float)
    if h is None:
        h = float(np.finfo(float).eps) ** (1 / 3)
    steps = h * (1.0 + np.abs(x))
    g = np.empty(x.size)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = steps[i]
        g[i] = (f(x + e) - f(x - e)) / (2.0 * steps[i])
    return g


def finite_difference_jacobian(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float | None = None,
) -> np.ndarray:
    """Approximate the Jacobian of a vector-valued ``fun`` at ``x``.

    The same central-difference scheme as
    :func:`finite_difference_gradient`, applied to every component at once,
    so the whole ``(m, n)`` Jacobian costs ``2 n`` evaluations of ``fun``
    rather than ``2 n`` per row.

    Parameters
    ----------
    fun : callable
        Vector-valued function of a 1-D array, returning shape (m,).
    x : np.ndarray, shape (n,)
        Point at which to differentiate.
    h : float, optional
        Base step size; see :func:`finite_difference_gradient`.

    Returns
    -------
    np.ndarray, shape (m, n)
        Jacobian estimate, row ``i`` being the gradient of component ``i``.
    """
    x = np.asarray(x, dtype=float)
    if h is None:
        h = float(np.finfo(float).eps) ** (1 / 3)
    steps = h * (1.0 + np.abs(x))
    columns = []
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = steps[i]
        forward = np.atleast_1d(np.asarray(fun(x + e), dtype=float))
        backward = np.atleast_1d(np.asarray(fun(x - e), dtype=float))
        columns.append((forward - backward) / (2.0 * steps[i]))
    return np.column_stack(columns)
