"""Nonlinear programming problem description."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mopt.nonlinear.finite_diff import finite_difference_gradient


@dataclass
class NLPProblem:
    """An unconstrained nonlinear program.

    Describes the problem

    .. math::

        \\min_{x \\in \\mathbb{R}^n} \\; f(x)

    Attributes
    ----------
    f : callable
        Objective ``f(x) -> float`` for a 1-D array ``x``.
    x0 : np.ndarray, shape (n,)
        Starting point; also fixes the problem dimension.
    grad : callable or None
        Gradient ``grad(x) -> np.ndarray`` of shape (n,). When None,
        :meth:`gradient` falls back to central finite differences of ``f``.
    hess : callable or None
        Hessian ``hess(x) -> np.ndarray`` of shape (n, n). Optional; needed
        only by second-order solvers such as Newton's method.
    """

    f: Callable[[np.ndarray], float]
    x0: np.ndarray
    grad: Callable[[np.ndarray], np.ndarray] | None = None
    hess: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self):
        if not callable(self.f):
            raise TypeError("f must be callable.")
        self.x0 = np.atleast_1d(np.asarray(self.x0, dtype=float))
        if self.x0.ndim != 1:
            raise ValueError("x0 must be a 1-D array.")

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of ``f`` at ``x``.

        Uses the user-supplied ``grad`` when given, otherwise central
        finite differences of ``f``. Solvers should call this rather than
        ``grad`` directly so the fallback applies uniformly.
        """
        if self.grad is not None:
            return np.asarray(self.grad(x), dtype=float)
        return finite_difference_gradient(self.f, x)
