"""Nonlinear programming problem descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mopt.nonlinear.finite_diff import (
    finite_difference_gradient,
    finite_difference_jacobian,
)


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


@dataclass
class ConstrainedNLPProblem(NLPProblem):
    """Minimize ``f`` over a closed convex set described by oracles.

    Describes the problem

    .. math::

        \\min_{x \\in S} \\; f(x), \\qquad S \\subseteq \\mathbb{R}^n
        \\text{ closed and convex.}

    Feasible-set-aware solvers never need an algebraic description of
    ``S``, only one of two oracles, so that is what this carries. Which
    one you must supply depends on the solver:
    :class:`~mopt.nonlinear.ProjectedGradient` needs ``projection``,
    :class:`~mopt.nonlinear.FrankWolfe` needs ``lmo``. Supplying both
    lets either solve the same problem; each raises ``ValueError`` when
    the oracle it needs is missing.

    Inherits ``f``, ``x0``, ``grad``, ``hess`` and
    :meth:`~NLPProblem.gradient` (with its finite-difference fallback)
    from :class:`NLPProblem`. Unconstrained solvers accept this type and
    simply ignore the extra fields, which is correct only if you intended
    to drop the constraints.

    Attributes
    ----------
    projection : callable or None
        ``projection(y) -> np.ndarray``, the point of ``S`` nearest ``y``.
        Must be the *exact* projection: solvers rely on the variational
        inequality it satisfies, and a merely feasible point breaks their
        descent guarantee. Build one with the factories in
        :mod:`mopt.nonlinear.projected_gradient`.
    lmo : callable or None
        ``lmo(grad) -> np.ndarray``, a linear minimization oracle
        returning :math:`\\arg\\min_{y \\in S} g^T y`. Requires ``S`` to
        be bounded, or the minimum is unbounded and the value undefined.
    ineq, eq : callable or None
        Optional *algebraic* description of the same set,
        ``ineq(x) -> np.ndarray`` feasible where every entry is ``<= 0``
        and ``eq(x) -> np.ndarray`` feasible where every entry is ``0``.
        The solvers never use these — an oracle is enough to optimize —
        but Lagrange multipliers attach to individual constraints, so
        :mod:`~mopt.nonlinear.constrained.optimality` cannot work without
        them.
    ineq_jac, eq_jac : callable or None
        Jacobians of ``ineq`` and ``eq``, shape (m, n) and (p, n). When
        None, central finite differences of the corresponding function are
        used.
    """

    projection: Callable[[np.ndarray], np.ndarray] | None = None
    lmo: Callable[[np.ndarray], np.ndarray] | None = None
    ineq: Callable[[np.ndarray], np.ndarray] | None = None
    eq: Callable[[np.ndarray], np.ndarray] | None = None
    ineq_jac: Callable[[np.ndarray], np.ndarray] | None = None
    eq_jac: Callable[[np.ndarray], np.ndarray] | None = None

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project ``x`` onto the feasible set.

        Raises
        ------
        ValueError
            If no ``projection`` oracle was supplied.
        """
        if self.projection is None:
            raise ValueError("This problem has no projection oracle.")
        return np.asarray(self.projection(x), dtype=float)

    def ineq_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of ``ineq`` at ``x``, shape (m, n).

        Uses ``ineq_jac`` when given, otherwise central finite differences.
        Returns an empty ``(0, n)`` array when the problem has no
        inequality constraints, so callers can stack unconditionally.
        """
        return self._jacobian(self.ineq, self.ineq_jac, x)

    def eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of ``eq`` at ``x``, shape (p, n).

        Uses ``eq_jac`` when given, otherwise central finite differences.
        Returns an empty ``(0, n)`` array when the problem has no equality
        constraints.
        """
        return self._jacobian(self.eq, self.eq_jac, x)

    @staticmethod
    def _jacobian(
        fun: Callable | None,
        jac: Callable | None,
        x: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if fun is None:
            return np.zeros((0, x.size))
        if jac is not None:
            return np.atleast_2d(np.asarray(jac(x), dtype=float))
        return np.atleast_2d(finite_difference_jacobian(fun, x))
