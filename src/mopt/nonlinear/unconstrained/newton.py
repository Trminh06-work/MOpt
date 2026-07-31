"""Newton's method (damped and undamped) for unconstrained problems."""

from __future__ import annotations

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.line_search import LineSearch, armijo
from mopt.nonlinear.problem import NLPProblem


class Newton(BaseOptimizer):
    """Newton's method for unconstrained problems.

    Each iteration solves the Newton system for the search direction and
    steps along it:

    .. math::

        H(x_k)\\, d_k = -\\nabla f(x_k), \\qquad
        x_{k+1} = x_k + \\eta_k d_k.

    *Damped* Newton picks :math:`\\eta_k` with a line search, which
    safeguards progress far from the minimizer (for descent Newton
    directions). *Undamped* (pure) Newton always takes the unit step
    :math:`\\eta_k = 1`: near a minimizer with positive definite Hessian it
    converges quadratically, but it has no global safeguards — it may
    diverge from poor starting points and follows the Newton direction
    even toward saddle points.

    Requires ``problem.hess``; the gradient comes from
    ``problem.gradient`` (user-supplied or finite-difference fallback).

    Parameters
    ----------
    damped : bool
        Use a line search for the step size (True) or the pure unit step
        (False).
    line_search : LineSearch
        Step-size rule for the damped variant, invoked as
        ``line_search(f, x, d, grad_f)`` with the Newton direction ``d``.
        Ignored when ``damped=False``. Note the ``eta=1.0`` default of the
        built-in searches tries the pure Newton step first.
    tol : float
        Convergence threshold on the gradient norm
        :math:`\\lVert \\nabla f(x) \\rVert_2`.
    max_iter : int
        Maximum number of Newton steps.
    """

    def __init__(
        self,
        damped: bool = True,
        line_search: LineSearch = armijo,
        tol: float = 1e-8,
        max_iter: int = 100,
    ):
        self.damped = damped
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        if problem.hess is None:
            raise ValueError("Newton's method requires problem.hess.")
        x = problem.x0.copy()
        for k in range(self.max_iter):
            g = problem.gradient(x)
            if np.linalg.norm(g) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: gradient norm below tol.", n_iter=k,
                )
            H = np.asarray(problem.hess(x), dtype=float)
            try:
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Singular Hessian at iteration {k}.", n_iter=k,
                )
            if self.damped:
                try:
                    _, eta = self.line_search(problem.f, x, d, problem.gradient)
                except ValueError:
                    return OptimizeResult(
                        x=x, fun=float(problem.f(x)), success=False,
                        message="Newton direction is not a descent direction "
                                "(Hessian not positive definite).", n_iter=k,
                    )
                except RuntimeError as exc:
                    return OptimizeResult(
                        x=x, fun=float(problem.f(x)), success=False,
                        message=f"Line search failed: {exc}", n_iter=k,
                    )
            else:
                eta = 1.0
            x = x + eta * d
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )
