"""Steepest-descent solver with a pluggable line search."""

from __future__ import annotations

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.line_search import LineSearch, armijo
from mopt.nonlinear.problem import NLPProblem


class GradientDescent(BaseOptimizer):
    """Gradient (steepest) descent for unconstrained problems.

    At each iterate the search direction is the negative gradient,

    .. math::

        d_k = -\\nabla f(x_k), \\qquad x_{k+1} = x_k + \\eta_k d_k,

    with the step size :math:`\\eta_k` chosen by the configured line
    search. Converges linearly on smooth convex problems; the rate
    degrades with the conditioning of the Hessian.

    Parameters
    ----------
    line_search : LineSearch
        Step-size rule invoked as ``line_search(f, x, d, grad_f)``.
        Defaults to :func:`~mopt.nonlinear.armijo`; configure tunables via
        ``functools.partial``, e.g. ``partial(wolfe, types="weak")``.
    tol : float
        Convergence threshold on the gradient norm
        :math:`\\lVert \\nabla f(x) \\rVert_2`.
    max_iter : int
        Maximum number of descent steps.
    """

    def __init__(
        self,
        line_search: LineSearch = armijo,
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ):
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        x = problem.x0.copy()
        for k in range(self.max_iter):
            g = problem.gradient(x)
            if np.linalg.norm(g) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: gradient norm below tol.", n_iter=k,
                )
            try:
                _, eta = self.line_search(problem.f, x, -g, problem.gradient)
            except RuntimeError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Line search failed: {exc}", n_iter=k,
                )
            x = x - eta * g
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )
