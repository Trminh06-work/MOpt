"""Quasi-Newton methods for unconstrained minimization.

Every method here replaces the Newton system with a secant approximation
of the inverse Hessian, built from the step and the gradient change alone:

.. math::

    d_k = -H_k \\nabla f(x_k), \\qquad H_{k+1} y_k = s_k, \\qquad
    s_k = x_{k+1} - x_k, \\quad y_k = \\nabla f(x_{k+1}) - \\nabla f(x_k).

This buys a superlinear rate without ever forming :math:`\\nabla^2 f`,
placing the family between gradient descent (linear, gradient only) and
Newton (quadratic, Hessian required). The secant equation alone leaves
:math:`H_{k+1}` underdetermined — it is ``n`` equations for the
``n(n+1)/2`` free entries of a symmetric matrix — so the variants differ
in which correction they add to :math:`H_k`. :func:`dfp` and :func:`bfgs`
are the two rank-2 choices, duals of each other under
:math:`s \\leftrightarrow y`, :math:`H \\leftrightarrow B`. See Grippo &
Sciandrone, and Andrei, *Modern Numerical Nonlinear Optimization*.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.line_search import LineSearch, bracketing_wolfe
from mopt.nonlinear.problem import NLPProblem


class QuasiNewtonUpdate(Protocol):
    """Call contract for quasi-Newton inverse-Hessian updates.

    Given the current approximation ``H``, the step ``s`` and the gradient
    change ``y``, return the updated approximation, which must satisfy the
    secant equation ``H_next @ y == s``. Implementations may assume the
    curvature condition ``s @ y > 0`` holds (the solver checks it first) —
    that is what lets them keep a positive definite ``H`` positive
    definite, and it is also what makes their denominators nonzero.
    """

    def __call__(
        self,
        H: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray: ...


def dfp(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Davidon-Fletcher-Powell update.

    .. math::

        H_{k+1} = H_k + \\frac{s_k s_k^T}{s_k^T y_k}
                      - \\frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k}

    The rank-2 correction that solves the inverse secant equation with
    :math:`u = s_k` and :math:`v = H_k y_k`. Historically the first
    quasi-Newton formula; it is the dual of :func:`bfgs` but recovers less
    well from a poor approximation, because an over-large :math:`H_k`
    persists across iterations rather than being corrected.

    Parameters
    ----------
    H : np.ndarray, shape (n, n)
        Current inverse-Hessian approximation; symmetric positive definite.
    s : np.ndarray, shape (n,)
        Step :math:`x_{k+1} - x_k`.
    y : np.ndarray, shape (n,)
        Gradient change :math:`\\nabla f(x_{k+1}) - \\nabla f(x_k)`; must
        satisfy ``s @ y > 0``.

    Returns
    -------
    np.ndarray, shape (n, n)
        The updated approximation.
    """
    Hy = H @ y
    return H + np.outer(s, s) / float(s @ y) - np.outer(Hy, Hy) / float(y @ Hy)


def bfgs(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Broyden-Fletcher-Goldfarb-Shanno update, in inverse form.

    BFGS is stated as the *direct* update of :math:`B_k \\approx
    \\nabla^2 f`,

    .. math::

        B_{k+1} = B_k - \\frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}
                      + \\frac{y_k y_k^T}{y_k^T s_k},

    with :math:`u = y_k` and :math:`v = B_k s_k`. Inverting it with the
    Sherman-Morrison-Woodbury formula gives the form used here, which
    needs no linear solve per iteration:

    .. math::

        H_{k+1} = (I - \\rho_k s_k y_k^T) H_k (I - \\rho_k y_k s_k^T)
                  + \\rho_k s_k s_k^T, \\qquad \\rho_k = 1 / (y_k^T s_k)

    The default update, and the standard choice in practice: it
    self-corrects a bad approximation within a few iterations, where
    :func:`dfp` may not.

    Parameters
    ----------
    H : np.ndarray, shape (n, n)
        Current inverse-Hessian approximation; symmetric positive definite.
    s : np.ndarray, shape (n,)
        Step :math:`x_{k+1} - x_k`.
    y : np.ndarray, shape (n,)
        Gradient change :math:`\\nabla f(x_{k+1}) - \\nabla f(x_k)`; must
        satisfy ``s @ y > 0``.

    Returns
    -------
    np.ndarray, shape (n, n)
        The updated approximation.
    """
    rho = 1.0 / float(y @ s)
    V = np.eye(s.size) - rho * np.outer(s, y)
    return V @ H @ V.T + rho * np.outer(s, s)


class QuasiNewton(BaseOptimizer):
    """Quasi-Newton method with a pluggable rank-2 update.

    One unified scheme covers both variants, which differ only in the
    update applied at the end of each iteration:

    .. math::

        d_k = -H_k \\nabla f(x_k), \\qquad
        x_{k+1} = x_k + \\eta_k d_k, \\qquad
        H_{k+1} = \\texttt{update}(H_k, s_k, y_k),

    starting from :math:`H_0 = I`. Needs no Hessian, and every iteration
    costs matrix-vector work only.

    The direction is a descent direction whenever :math:`H_k` is positive
    definite, since :math:`\\nabla f^T d_k = -\\nabla f^T H_k \\nabla f <
    0`. Both updates preserve positive definiteness provided the curvature
    condition :math:`s_k^T y_k > 0` holds, which the Wolfe curvature
    condition guarantees; a run where it fails numerically stops and
    reports failure rather than updating into an indefinite ``H``.

    Parameters
    ----------
    update : QuasiNewtonUpdate
        Rank-2 rule invoked as ``update(H, s, y)``. Defaults to
        :func:`bfgs`; see :func:`dfp` for the dual formula.
    line_search : LineSearch
        Step-size rule invoked as ``line_search(f, x, d, grad_f)``.
        Defaults to :func:`~mopt.nonlinear.bracketing_wolfe`, whose
        curvature condition the positive definiteness of ``H`` depends on;
        configure tunables via ``functools.partial``. It must be able to
        *grow* the trial step: the acceptable window routinely sits above
        :math:`\\eta = 1` here, out of reach of the shrink-only
        :func:`~mopt.nonlinear.wolfe`. Note also that
        :func:`~mopt.nonlinear.armijo` alone does not imply
        :math:`s_k^T y_k > 0`.
    H0 : np.ndarray, optional
        Initial inverse-Hessian approximation, shape ``(n, n)`` and
        symmetric positive definite. None (the default) uses the identity,
        which makes the first step a steepest-descent step, and then
        rescales it to :math:`(s_0^T y_0 / y_0^T y_0) I` before the first
        update (Nocedal & Wright, eq. 6.20). That factor estimates the
        size of the true inverse Hessian along the first step, which is
        what puts the unit step :math:`\\eta = 1` in range — an identity
        ``H_0`` is scale-blind, and the well-scaled step can be orders of
        magnitude away from 1, out of reach of a backtracking search. An
        explicitly supplied ``H0`` is used as given, never rescaled.
    tol : float
        Convergence threshold on the gradient norm
        :math:`\\lVert \\nabla f(x) \\rVert_2`.
    max_iter : int
        Maximum number of iterations.
    """

    def __init__(
        self,
        update: QuasiNewtonUpdate = bfgs,
        line_search: LineSearch = bracketing_wolfe,
        H0: np.ndarray | None = None,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.update = update
        self.line_search = line_search
        self.H0 = H0
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        x = problem.x0.copy()
        if self.H0 is None:
            H = np.eye(x.size)
        else:
            H = np.asarray(self.H0, dtype=float)
            if H.shape != (x.size, x.size):
                raise ValueError(
                    f"H0 has shape {H.shape}, expected {(x.size, x.size)}."
                )
        g = problem.gradient(x)
        for k in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: gradient norm below tol.", n_iter=k,
                )
            d = -H @ g
            try:
                _, eta = self.line_search(problem.f, x, d, problem.gradient)
            except ValueError:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Direction is not a descent direction at "
                            f"iteration {k} (H is not positive definite).",
                    n_iter=k,
                )
            except RuntimeError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Line search failed: {exc}", n_iter=k,
                )
            x_next = x + eta * d
            g_next = problem.gradient(x_next)
            s, y = x_next - x, g_next - g
            x, g = x_next, g_next  # the step itself is accepted either way
            if float(s @ y) <= 0:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Curvature condition s @ y > 0 failed at "
                            f"iteration {k}; the update would not preserve "
                            "positive definiteness.", n_iter=k + 1,
                )
            if k == 0 and self.H0 is None:
                H = (float(s @ y) / float(y @ y)) * H
            H = self.update(H, s, y)
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )
