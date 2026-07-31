"""Trust-region methods for unconstrained minimization.

At each iterate the objective is replaced by the quadratic model
``m(d) = f(x) + g @ d + 0.5 * d @ B @ d``, minimized approximately within
the ball ``||d|| <= radius`` by a pluggable subproblem method (Cauchy
point or dogleg). The ratio of actual to predicted reduction decides
whether the step is taken and how the radius evolves. See Nocedal &
Wright, chapter 4.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.problem import NLPProblem


class TrustRegionMethod(Protocol):
    """Call contract for trust-region subproblem solvers.

    Given the local model — gradient ``grad`` and symmetric model matrix
    ``hessian`` — and the current ``radius``, return a step ``d`` with
    ``||d|| <= radius`` that reduces the model. Implementations may assume
    ``grad`` is nonzero (the solver checks convergence first) and should
    raise ``ValueError`` or ``numpy.linalg.LinAlgError`` when the model
    violates their assumptions; the solver reports that as a failed run.
    """

    def __call__(
        self,
        grad: np.ndarray,
        hessian: np.ndarray,
        radius: float,
    ) -> np.ndarray: ...


def cauchy(grad: np.ndarray, hessian: np.ndarray, radius: float) -> np.ndarray:
    """Cauchy point: minimize the model along steepest descent.

    With ``d_s = -radius * g / ||g||`` the full step to the boundary, the
    model along ``alpha * d_s`` is minimized at

    .. math::

        \\alpha = \\min\\!\\left(1,
            \\frac{\\lVert g\\rVert^3}{\\Delta\\, g^T B g}\\right),

    and for nonpositive curvature (``g @ B @ g <= 0``) the model decreases
    all the way to the boundary, so ``alpha = 1``. Works for any symmetric
    ``hessian``; convergence is steepest-descent-like (first order).

    Parameters
    ----------
    grad : np.ndarray, shape (n,)
        Gradient of the objective at the current iterate; must be nonzero.
    hessian : np.ndarray, shape (n, n)
        Symmetric model matrix (exact Hessian or an approximation).
    radius : float
        Trust-region radius.

    Returns
    -------
    np.ndarray, shape (n,)
        The Cauchy step, with norm at most ``radius``.
    """
    norm_g = np.linalg.norm(grad)
    d_s = -radius * grad / norm_g
    a = grad @ hessian @ grad
    alpha = 1.0 if a <= 0 else min(1.0, norm_g**3 / (radius * a))
    return alpha * d_s


def dogleg(grad: np.ndarray, hessian: np.ndarray, radius: float) -> np.ndarray:
    """Dogleg step; requires a positive definite model matrix.

    The dogleg path runs from 0 to the unconstrained steepest-descent
    minimizer ``d_sd = -(||g||^2 / g@B@g) g`` and on to the Newton step
    ``d_n = -B^{-1} g``. Return the Newton step when it fits inside the
    region, otherwise the point where the path crosses the boundary.
    Combines Newton-like convergence near the solution with safe short
    steps far from it.

    Parameters
    ----------
    grad : np.ndarray, shape (n,)
        Gradient of the objective at the current iterate; must be nonzero.
    hessian : np.ndarray, shape (n, n)
        Symmetric positive definite model matrix.
    radius : float
        Trust-region radius.

    Returns
    -------
    np.ndarray, shape (n,)
        The dogleg step, with norm at most ``radius``.

    Raises
    ------
    ValueError
        If the curvature along the gradient is nonpositive (the model
        matrix is not positive definite).
    numpy.linalg.LinAlgError
        If the model matrix is singular.
    """
    a = grad @ hessian @ grad
    if a <= 0:
        raise ValueError(
            "dogleg requires positive curvature along the gradient "
            "(positive definite model matrix)"
        )
    d_newton = -np.linalg.solve(hessian, grad)
    if np.linalg.norm(d_newton) <= radius:
        return d_newton
    d_sd = -(np.linalg.norm(grad) ** 2 / a) * grad
    if np.linalg.norm(d_sd) >= radius:
        return radius * d_sd / np.linalg.norm(d_sd)
    # second dogleg segment: solve ||d_sd + t*(d_newton - d_sd)|| = radius,
    # a quadratic in t with exactly one root in (0, 1)
    v = d_newton - d_sd
    uv, vv, uu = d_sd @ v, v @ v, d_sd @ d_sd
    t = (-uv + np.sqrt(uv**2 + vv * (radius**2 - uu))) / vv
    return d_sd + t * v


class TrustRegion(BaseOptimizer):
    """Trust-region solver for :class:`NLPProblem`.

    Each iteration builds the quadratic model from ``problem.gradient``
    and ``problem.hess``, asks ``method`` for a step within the current
    radius, and computes the acceptance ratio

    .. math::

        \\rho = \\frac{f(x) - f(x + d)}{m(0) - m(d)}

    of actual to predicted reduction. The step is taken when
    ``rho > acp_ratio_lb``; the radius halves when the model fit is poor
    (``rho <= acp_ratio_lb``) and doubles when it is good
    (``rho >= acp_ratio_ub``) *and* the step was boundary-limited.

    Requires ``problem.hess``; the gradient comes from
    ``problem.gradient`` (user-supplied or finite-difference fallback).

    Parameters
    ----------
    method : TrustRegionMethod
        Subproblem solver invoked as ``method(grad, hessian, radius)``.
        Defaults to :func:`cauchy` (robust, first order);
        :func:`dogleg` converges much faster but requires a positive
        definite Hessian.
    radius : float
        Initial trust-region radius.
    tol : float
        Convergence threshold on the gradient norm.
    radius_tol : float
        Give up when the radius shrinks below this (the model is
        persistently poor at the current iterate).
    max_iter : int
        Maximum number of outer iterations.
    acp_ratio_lb, acp_ratio_ub : float
        Acceptance-ratio thresholds controlling step acceptance and
        radius updates, ``0 < acp_ratio_lb < acp_ratio_ub < 1``.
    """

    def __init__(
        self,
        method: TrustRegionMethod = cauchy,
        radius: float = 1.0,
        tol: float = 1e-6,
        radius_tol: float = 1e-3,
        max_iter: int = 100,
        acp_ratio_lb: float = 0.25,
        acp_ratio_ub: float = 0.75,
    ):
        self.method = method
        self.radius = radius
        self.tol = tol
        self.radius_tol = radius_tol
        self.max_iter = max_iter
        self.acp_ratio_lb = acp_ratio_lb
        self.acp_ratio_ub = acp_ratio_ub

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        if problem.hess is None:
            raise ValueError("Trust-region methods require problem.hess.")
        x = problem.x0.copy()
        radius = self.radius
        num_iter = 0
        while num_iter < self.max_iter and radius > self.radius_tol:
            g = problem.gradient(x)
            if np.linalg.norm(g) < self.tol:
                break
            B = np.asarray(problem.hess(x), dtype=float)
            try:
                d = self.method(g, B, radius)
            except (ValueError, np.linalg.LinAlgError) as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Subproblem failed: {exc}", n_iter=num_iter,
                )

            # acceptance ratio; m(0) - m(d) = -(g @ d + 0.5 d @ B @ d)
            predicted = -(g @ d + 0.5 * (d @ B @ d))
            if predicted <= 0:  # no predicted decrease: distrust the model
                radius /= 2
                num_iter += 1
                continue
            acp_ratio = (problem.f(x) - problem.f(x + d)) / predicted
            if acp_ratio > self.acp_ratio_lb:
                x = x + d

            # update radius; grow only when the step was boundary-limited
            if acp_ratio <= self.acp_ratio_lb:
                radius /= 2
            elif (
                acp_ratio >= self.acp_ratio_ub
                and np.linalg.norm(d) >= radius * (1 - 1e-10)
            ):
                radius *= 2
            num_iter += 1

        fun = float(problem.f(x))
        if np.linalg.norm(problem.gradient(x)) < self.tol:
            return OptimizeResult(
                x=x, fun=fun, success=True,
                message="Converged: gradient norm below tol.", n_iter=num_iter,
            )
        if radius <= self.radius_tol:
            return OptimizeResult(
                x=x, fun=fun, success=False,
                message="Trust-region radius below tolerance.", n_iter=num_iter,
            )
        return OptimizeResult(
            x=x, fun=fun, success=False,
            message="Iteration limit reached.", n_iter=num_iter,
        )
