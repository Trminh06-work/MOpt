"""Feasible-direction methods for minimization over a convex set.

Both solvers here keep every iterate inside the feasible set ``S`` and
differ only in the oracle they ask for. :class:`ProjectedGradient` takes an
unconstrained gradient step and projects the result back onto ``S``;
:class:`FrankWolfe` never projects, and instead minimizes the linearization
of ``f`` over ``S`` to pick a point to move toward. Both then take a step of
length at most 1 along ``d``, which preserves feasibility for free: with
``x`` and the target both in ``S``, ``x + eta * d`` is a convex combination
of two feasible points.

The module also builds the oracles. Each is a *factory* that does the
expensive setup (an eigendecomposition, a pseudoinverse) once and returns a
closure, so the per-projection cost is quadratic rather than cubic in ``n``.
See Nocedal & Wright chapter 12, and Bertsekas, *Nonlinear Programming*,
chapter 2.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.line_search import LineSearch, armijo
from mopt.nonlinear.problem import ConstrainedNLPProblem


class Projection(Protocol):
    """Call contract for projections onto a closed convex set.

    Given any point ``y``, return the point of the set nearest ``y`` in the
    Euclidean norm. Implementations must return the *exact* projection: the
    solvers here depend on the variational inequality
    ``(y - P(y)) @ (z - P(y)) <= 0`` for all feasible ``z``, which merely
    feasible points do not satisfy.
    """

    def __call__(self, y: np.ndarray) -> np.ndarray: ...


class LinearOracle(Protocol):
    """Call contract for linear minimization over a compact convex set.

    Given a gradient ``grad``, return a minimizer of the linear function
    ``grad @ y`` over the set. The set must be bounded for this to be
    well defined.
    """

    def __call__(self, grad: np.ndarray) -> np.ndarray: ...


def halfspace_projection(a: np.ndarray, b: float) -> Projection:
    """Projection onto the halfspace ``{y : a @ y <= b}``.

    KKT for ``min ||y - x||^2 / 2`` subject to ``a @ y <= b`` gives
    ``y = x - lam * a`` with ``lam >= 0`` and ``lam * (a @ y - b) = 0``,
    so the multiplier is available in closed form and complementary
    slackness collapses to a clamp:

    .. math::

        P(x) = x - \\max\\!\\left(0, \\frac{a^Tx - b}{a^Ta}\\right) a.

    Parameters
    ----------
    a : np.ndarray, shape (n,)
        Halfspace normal; must be nonzero.
    b : float
        Halfspace offset.

    Returns
    -------
    Projection
        Callable mapping a point to its projection.

    Raises
    ------
    ValueError
        If ``a`` is the zero vector.
    """
    a = np.asarray(a, dtype=float)
    aa = float(a @ a)
    if aa == 0.0:
        raise ValueError("a must be a nonzero vector.")

    def project(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return y - max(0.0, (float(a @ y) - b) / aa) * a

    return project


def affine_projection(A: np.ndarray, b: np.ndarray) -> Projection:
    """Projection onto the affine set ``{y : A @ y == b}``.

    The same KKT form as :func:`halfspace_projection`, but an equality
    multiplier is free in sign, so there is no clamp:

    .. math::

        P(x) = x - A^{+}(Ax - b),

    with ``A+`` the pseudoinverse, which equals ``A^T (A A^T)^{-1}`` when
    ``A`` has full row rank. The pseudoinverse is formed once here, so each
    projection costs one matrix-vector product per factor.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Constraint matrix; ``m <= n``, ideally full row rank. A rank
        deficient ``A`` is tolerated as long as the system is consistent,
        since the pseudoinverse then gives the least-norm correction.
    b : np.ndarray, shape (m,)
        Right-hand side.

    Returns
    -------
    Projection
        Callable mapping a point to its projection.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if A.shape[0] != b.size:
        raise ValueError(
            f"A has {A.shape[0]} rows but b has {b.size} entries."
        )
    A_pinv = np.linalg.pinv(A)  # formed once, reused across calls

    def project(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return y - A_pinv @ (A @ y - b)

    return project


def ellipsoid_projection(
    M: np.ndarray,
    s: float,
    max_bisect: int = 100,
    tol: float = 1e-12,
) -> Projection:
    """Projection onto the ellipsoid ``{y : y @ M @ y <= s}``.

    The only projector here without a closed form, because the constraint
    gradient ``2 M y`` depends on ``y``. KKT gives
    ``y(lam) = (I + lam M)^{-1} x``, and the multiplier solves the scalar
    secular equation ``phi(lam) = y(lam) @ M @ y(lam) - s = 0``. In the
    eigenbasis ``M = V diag(mu) V^T``, with ``z = V^T x``,

    .. math::

        \\phi(\\lambda)
            = \\sum_i \\frac{\\mu_i z_i^2}{(1 + \\lambda \\mu_i)^2} - s,

    which decreases strictly from ``x @ M @ x - s`` toward ``-s``, so the
    root is unique. Because
    ``phi(lam) + s <= (x @ M @ x) / (1 + lam * mu_min)^2`` the root is
    bracketed analytically, with no search for an upper bound, and
    bisection is then unconditionally safe.

    The eigendecomposition is computed once, so a projection costs two
    matrix-vector products plus ``max_bisect`` vector operations —
    ``O(n^2)``, against ``O(n^3)`` for re-solving ``(I + lam M) y = x`` at
    every bisection step.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Symmetric positive definite shape matrix.
    s : float
        Level, strictly positive.
    max_bisect : int
        Bisection steps. The bracket halves each step, so the default is
        far past machine precision for any reasonable conditioning.
    tol : float
        Relative tolerance on the constraint value. A point within
        ``tol * s`` of the boundary is treated as already feasible; this
        also absorbs the rounding that would otherwise collapse the
        bracket for a point sitting exactly on the boundary.

    Returns
    -------
    Projection
        Callable mapping a point to its projection.

    Raises
    ------
    ValueError
        If ``M`` is not symmetric positive definite, or ``s <= 0``.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")
    if not np.allclose(M, M.T):
        raise ValueError("M must be symmetric.")
    if s <= 0:
        raise ValueError("s must be positive.")
    mu, V = np.linalg.eigh(M)  # ascending; computed once
    if mu[0] <= 0:
        raise ValueError("M must be positive definite.")
    mu_min = float(mu[0])

    def project(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = V.T @ y
        weights = mu * z**2

        def phi(lam: float) -> float:
            return float(np.sum(weights / (1.0 + lam * mu) ** 2)) - s

        phi_0 = phi(0.0)
        if phi_0 <= tol * s:  # inactive, or already on the boundary
            return y.copy()

        lo, hi = 0.0, (np.sqrt(1.0 + phi_0 / s) - 1.0) / mu_min
        for _ in range(max_bisect):  # phi(lo) > 0 >= phi(hi) throughout
            mid = 0.5 * (lo + hi)
            if phi(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return V @ (z / (1.0 + 0.5 * (lo + hi) * mu))

    return project


def intersection_projection(
    projections: Sequence[Projection],
    tol: float = 1e-14,
    max_iter: int = 2000,
) -> Projection:
    """Projection onto an intersection, by Dykstra's algorithm.

    Projections do not compose: ``P_B(P_A(x))`` lands in ``A n B`` but is
    generally not the nearest such point. Dykstra carries one correction
    vector per set, added back before that set's projection, and converges
    to the true ``P_{A n B}``.

    The correction vectors are what make this usable here. Dropping them
    gives plain alternating projections (POCS), which is exact only when
    every set is affine; with a curved set in the mix it returns a merely
    feasible point, and :class:`ProjectedGradient` then loses its descent
    guarantee and can step uphill.

    Parameters
    ----------
    projections : sequence of Projection
        Projections onto the individual sets, which must have a nonempty
        intersection.
    tol : float
        Stop once an entire sweep moves the iterate less than this. It
        bounds the accuracy of the result, and so also the smallest useful
        ``tol`` for a solver built on top.
    max_iter : int
        Maximum number of sweeps.

    Returns
    -------
    Projection
        Callable mapping a point to its projection onto the intersection.

    References
    ----------
    Boyle & Dykstra (1986); Bauschke & Borwein, *SIAM Review* 38(3), 1996.
    https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm
    """
    projections = list(projections)
    if not projections:
        raise ValueError("projections must not be empty.")

    def project(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).copy()
        corrections = [np.zeros_like(y) for _ in projections]
        for _ in range(max_iter):
            y_prev = y.copy()
            for i, project_i in enumerate(projections):
                shifted = y + corrections[i]
                y = np.asarray(project_i(shifted), dtype=float)
                corrections[i] = shifted - y  # the only difference from POCS
            if np.linalg.norm(y - y_prev) <= tol:
                break
        return y

    return project


def ellipsoid_oracle(M: np.ndarray, s: float) -> LinearOracle:
    """Linear minimization oracle for ``{y : y @ M @ y <= s}``.

    A linear function attains its minimum over an ellipsoid on the
    boundary, along ``-M^{-1} g`` scaled to touch it:

    .. math::

        \\arg\\min_{y^T M y \\le s} g^T y
            = -\\sqrt{\\frac{s}{g^T M^{-1} g}}\\, M^{-1} g.

    ``M`` is eigendecomposed once, so each call costs two matrix-vector
    products.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Symmetric positive definite shape matrix.
    s : float
        Level, strictly positive.

    Returns
    -------
    LinearOracle
        Callable mapping a gradient to a minimizer over the ellipsoid.

    Raises
    ------
    ValueError
        If ``M`` is not symmetric positive definite, or ``s <= 0``.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")
    if not np.allclose(M, M.T):
        raise ValueError("M must be symmetric.")
    if s <= 0:
        raise ValueError("s must be positive.")
    mu, V = np.linalg.eigh(M)
    if mu[0] <= 0:
        raise ValueError("M must be positive definite.")

    def lmo(grad: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad, dtype=float)
        w = V @ ((V.T @ grad) / mu)  # M^{-1} grad
        denom = float(grad @ w)
        if denom <= 0:  # grad = 0: every point is optimal, so stay put
            return np.zeros_like(grad)
        return -np.sqrt(s / denom) * w

    return lmo


class ProjectedGradient(BaseOptimizer):
    """Projected gradient descent over a convex set.

    Each iteration throws a trial point ``alpha`` along the negative
    gradient, projects it back, and line searches along the resulting
    feasible direction:

    .. math::

        \\hat{x} = P_S(x_k - \\alpha \\nabla f(x_k)), \\qquad
        d_k = \\hat{x} - x_k, \\qquad
        x_{k+1} = x_k + \\eta_k d_k.

    The variational inequality characterizing ``P_S`` gives
    :math:`\\nabla f(x)^T d \\le -\\lVert d \\rVert^2 / \\alpha`, so ``d``
    is a descent direction whenever ``x`` is not already stationary, and
    :math:`\\lVert d \\rVert` is the natural stationarity measure — it
    vanishes exactly at a constrained stationary point, which is what the
    solver tests. Feasibility needs no extra care because ``eta <= 1``
    makes each iterate a convex combination of ``x_k`` and ``x_hat``.

    Requires ``problem.projection``.

    Parameters
    ----------
    alpha : float
        Projection step. Distinct from the line-search step ``eta``: it
        sets how far the trial point is thrown *before* projecting.
        ``alpha ~ 1/L`` for an ``L``-smooth objective is the standard
        choice and is typically far faster than the default.
    line_search : LineSearch
        Step rule invoked as ``line_search(f, x, d, grad_f)``. Must return
        ``eta <= 1``, or iterates can leave the feasible set; the default
        :func:`~mopt.nonlinear.armijo` starts at 1 and only shrinks, so it
        is safe. Configure tunables with ``functools.partial``, keeping
        ``eta <= 1``.
    tol : float
        Convergence threshold on :math:`\\lVert d \\rVert`. Cannot usefully
        be pushed below the projection's own accuracy: the descent margin
        is only :math:`\\lVert d \\rVert^2 / \\alpha`, so at
        ``tol = 1e-7`` it is around ``1e-14``, which projection error
        swamps — ``d`` then reads as an ascent direction and the run stops
        early.
    max_iter : int
        Maximum number of iterations.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        line_search: LineSearch = armijo,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = alpha
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: ConstrainedNLPProblem) -> OptimizeResult:
        if getattr(problem, "projection", None) is None:
            raise ValueError("ProjectedGradient requires problem.projection.")
        x = problem.project(problem.x0)  # start feasible
        for k in range(self.max_iter):
            g = problem.gradient(x)
            d = problem.project(x - self.alpha * g) - x
            if np.linalg.norm(d) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: projected gradient below tol.",
                    n_iter=k,
                )
            try:
                _, eta = self.line_search(problem.f, x, d, problem.gradient)
            except ValueError:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Projected direction is not a descent direction "
                            f"at iteration {k}; the projection is likely "
                            "inexact, or tol is below its accuracy.",
                    n_iter=k,
                )
            except RuntimeError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Line search failed: {exc}", n_iter=k,
                )
            x = x + eta * d
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )


class FrankWolfe(BaseOptimizer):
    """Frank-Wolfe (conditional gradient) over a compact convex set.

    Never projects. Each iteration minimizes the linearization of ``f`` at
    the current iterate over the whole feasible set and steps toward the
    minimizer:

    .. math::

        y_k = \\arg\\min_{y \\in S} \\nabla f(x_k)^T y, \\qquad
        d_k = y_k - x_k, \\qquad
        x_{k+1} = x_k + \\eta_k d_k.

    That makes it the method of choice when the linear oracle is cheap but
    projection is not — over a simplex, an L1 ball or a nuclear-norm ball,
    for instance. The price is a sublinear ``O(1/k)`` rate: it zigzags when
    the solution lies on the boundary, so it reaches a given accuracy in
    ``f`` far sooner than the same accuracy in ``x``.

    The quantity :math:`-\\nabla f(x)^T d \\ge 0` is the **Frank-Wolfe
    gap**, an upper bound on :math:`f(x) - f^\\star`, so it doubles as a
    certificate and is what the solver tests for convergence.

    Requires ``problem.lmo``. ``problem.x0`` must be feasible; it is
    projected first when ``problem.projection`` is also supplied.

    Parameters
    ----------
    line_search : LineSearch
        Step rule invoked as ``line_search(f, x, d, grad_f)``. Must return
        ``eta <= 1``, or iterates can leave the feasible set; the default
        :func:`~mopt.nonlinear.armijo` starts at 1 and only shrinks, so it
        is safe.
    tol : float
        Convergence threshold on the Frank-Wolfe gap.
    max_iter : int
        Maximum number of iterations.

    Notes
    -----
    Frank-Wolfe expects the solution on the boundary. When no constraint is
    active the oracle keeps returning far-away boundary points, so ``d``
    stays long while the gradient shrinks, and the accepted step has to go
    to zero. A backtracking search can then run out of shrinks and the run
    stops with "Line search failed" while ``fun`` is already accurate —
    give it more shrinks (``partial(armijo, max_iter=400)``) or use
    :class:`ProjectedGradient`, which has no such trouble with an interior
    optimum.
    """

    def __init__(
        self,
        line_search: LineSearch = armijo,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: ConstrainedNLPProblem) -> OptimizeResult:
        if getattr(problem, "lmo", None) is None:
            raise ValueError("FrankWolfe requires problem.lmo.")
        x = problem.x0.copy()
        if getattr(problem, "projection", None) is not None:
            x = problem.project(x)  # make sure we start feasible
        for k in range(self.max_iter):
            g = problem.gradient(x)
            d = np.asarray(problem.lmo(g), dtype=float) - x
            gap = -float(g @ d)  # bounds f(x) - f*
            if gap < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: Frank-Wolfe gap below tol.", n_iter=k,
                )
            try:
                _, eta = self.line_search(problem.f, x, d, problem.gradient)
            except ValueError:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Frank-Wolfe direction is not a descent direction "
                            f"at iteration {k}; the oracle is likely not "
                            "minimizing over the feasible set.", n_iter=k,
                )
            except RuntimeError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Line search failed: {exc}", n_iter=k,
                )
            x = x + eta * d
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )
