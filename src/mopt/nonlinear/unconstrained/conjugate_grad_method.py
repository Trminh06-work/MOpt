"""Conjugate gradient methods for unconstrained minimization.

Every method here builds its search direction from the current gradient
and the previous direction,

.. math::

    d_{k+1} = -g_{k+1} + \\beta_{k+1} d_k,

which keeps successive directions conjugate with respect to the problem
curvature while storing only a single extra vector. On a quadratic the
recurrence with the exact step reaches the minimizer in at most ``n``
iterations (:class:`QuadraticCG`); on a general objective it is paired
with a line search and a choice of :math:`\\beta` rule
(:class:`ConjugateGradient`, :class:`ArmijoModifiedCG`). See Nocedal &
Wright, chapter 5.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.line_search import LineSearch, wolfe
from mopt.nonlinear.problem import NLPProblem


class BetaRule(Protocol):
    """Call contract for conjugate gradient ``beta`` coefficients.

    Given the new gradient ``grad_next``, the previous gradient ``grad``
    and the previous direction ``d``, return the scalar that mixes ``d``
    into the next direction. Implementations may assume ``grad`` is
    nonzero (the solver checks convergence first).
    """

    def __call__(
        self,
        grad_next: np.ndarray,
        grad: np.ndarray,
        d: np.ndarray,
    ) -> float: ...


def fletcher_reeves(
    grad_next: np.ndarray,
    grad: np.ndarray,
    d: np.ndarray,
) -> float:
    """Fletcher-Reeves coefficient.

    .. math::

        \\beta_{k+1} =
            \\frac{\\lVert g_{k+1} \\rVert^2}{\\lVert g_k \\rVert^2}

    Paired with a strong Wolfe search using ``sigma < 1/2`` this
    guarantees every direction is a descent direction. The coefficient is
    always nonnegative, so a bad direction is never damped out — after a
    short step the method can stall taking many further short steps.

    Parameters
    ----------
    grad_next : np.ndarray, shape (n,)
        Gradient at the new iterate.
    grad : np.ndarray, shape (n,)
        Gradient at the previous iterate; must be nonzero.
    d : np.ndarray, shape (n,)
        Previous search direction. Unused; present for :class:`BetaRule`.

    Returns
    -------
    float
        The Fletcher-Reeves coefficient.
    """
    return float(grad_next @ grad_next) / float(grad @ grad)


def polak_ribiere(
    grad_next: np.ndarray,
    grad: np.ndarray,
    d: np.ndarray,
) -> float:
    """Polak-Ribiere-Polyak coefficient.

    .. math::

        \\beta_{k+1} =
            \\frac{g_{k+1}^T (g_{k+1} - g_k)}{\\lVert g_k \\rVert^2}

    Usually faster than :func:`fletcher_reeves` because a short step makes
    the numerator small, which resets the direction toward steepest
    descent. It may be negative, and is not guaranteed to give a descent
    direction under a Wolfe search — pair it with :class:`ArmijoModifiedCG`,
    which enforces sufficient descent directly, or truncate it (PRP+).

    Parameters
    ----------
    grad_next : np.ndarray, shape (n,)
        Gradient at the new iterate.
    grad : np.ndarray, shape (n,)
        Gradient at the previous iterate; must be nonzero.
    d : np.ndarray, shape (n,)
        Previous search direction. Unused; present for :class:`BetaRule`.

    Returns
    -------
    float
        The Polak-Ribiere-Polyak coefficient.
    """
    return float(grad_next @ (grad_next - grad)) / float(grad @ grad)


class QuadraticCG(BaseOptimizer):
    """Linear conjugate gradient, for quadratic objectives only.

    For :math:`f(x) = \\tfrac12 x^T Q x - c^T x` with ``Q`` symmetric
    positive definite, the exact minimizer along ``d`` is available in
    closed form,

    .. math::

        \\eta_k = \\frac{\\lVert g_k \\rVert^2}{d_k^T Q d_k},

    so no line search is needed, and the gradient updates by the
    recurrence :math:`g_{k+1} = g_k + \\eta_k Q d_k` at one matrix-vector
    product per iteration. In exact arithmetic this terminates at the
    minimizer within ``n`` iterations.

    Requires ``problem.hess``, which supplies ``Q`` and **must be
    constant** — the solver evaluates it once at ``problem.x0``. Passing a
    non-quadratic problem is not detected and yields a wrong answer; use
    :class:`ConjugateGradient` for those.

    Parameters
    ----------
    tol : float
        Convergence threshold on the gradient norm
        :math:`\\lVert \\nabla f(x) \\rVert_2`.
    max_iter : int or None
        Maximum number of iterations. None (the default) uses the problem
        dimension, the exact-arithmetic bound.
    """

    def __init__(self, tol: float = 1e-8, max_iter: int | None = None):
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        if problem.hess is None:
            raise ValueError("QuadraticCG requires problem.hess.")
        x = problem.x0.copy()
        Q = np.asarray(problem.hess(x), dtype=float)
        max_iter = x.size if self.max_iter is None else self.max_iter

        g = problem.gradient(x)
        d = -g
        for k in range(max_iter):
            if np.linalg.norm(g) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: gradient norm below tol.", n_iter=k,
                )
            Qd = Q @ d
            curvature = float(d @ Qd)
            if curvature <= 0:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Nonpositive curvature d @ Q @ d at iteration {k} "
                            "(Q is not positive definite).", n_iter=k,
                )
            gg = float(g @ g)
            eta = gg / curvature
            x = x + eta * d
            g_next = g + eta * Qd
            d = -g_next + (float(g_next @ g_next) / gg) * d
            g = g_next

        if np.linalg.norm(g) < self.tol:
            return OptimizeResult(
                x=x, fun=float(problem.f(x)), success=True,
                message="Converged: gradient norm below tol.", n_iter=max_iter,
            )
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=max_iter,
        )


class ConjugateGradient(BaseOptimizer):
    """Nonlinear conjugate gradient with a pluggable ``beta`` rule.

    Each iteration picks a step along the current direction with the
    configured line search, then extends the direction recurrence

    .. math::

        x_{k+1} = x_k + \\eta_k d_k, \\qquad
        d_{k+1} = -g_{k+1} + \\beta_{k+1} d_k,

    starting from :math:`d_0 = -g_0`. Needs no Hessian and stores only a
    handful of vectors, which makes it the usual choice when ``n`` is too
    large for Newton or trust-region methods.

    The direction is only guaranteed to be a descent direction for
    particular pairings of ``beta`` and line search — Fletcher-Reeves with
    strong Wolfe and ``sigma < 1/2`` is the standard safe one, and is the
    default here. A run that produces an ascent direction stops and
    reports failure rather than stepping uphill.

    Parameters
    ----------
    beta : BetaRule
        Coefficient rule invoked as ``beta(grad_next, grad, d)``. Defaults
        to :func:`fletcher_reeves`; see :func:`polak_ribiere` for the
        faster but unsafeguarded alternative.
    line_search : LineSearch
        Step-size rule invoked as ``line_search(f, x, d, grad_f)``.
        Defaults to :func:`~mopt.nonlinear.wolfe` (strong), whose
        curvature condition the descent guarantee depends on; configure
        tunables via ``functools.partial``.
    tol : float
        Convergence threshold on the gradient norm
        :math:`\\lVert \\nabla f(x) \\rVert_2`.
    max_iter : int
        Maximum number of iterations.
    """

    def __init__(
        self,
        beta: BetaRule = fletcher_reeves,
        line_search: LineSearch = wolfe,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.beta = beta
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        x = problem.x0.copy()
        g = problem.gradient(x)
        d = -g
        for k in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: gradient norm below tol.", n_iter=k,
                )
            try:
                _, eta = self.line_search(problem.f, x, d, problem.gradient)
            except ValueError:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Direction is not a descent direction at "
                            f"iteration {k}; this beta rule and line search "
                            "do not guarantee descent.", n_iter=k,
                )
            except RuntimeError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Line search failed: {exc}", n_iter=k,
                )
            x = x + eta * d
            g_next = problem.gradient(x)
            d = -g_next + self.beta(g_next, g, d) * d
            g = g_next
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )


class ArmijoModifiedCG(BaseOptimizer):
    """Nonlinear CG under the Armijo Modified (AM) line search.

    AM backtracks like :func:`~mopt.nonlinear.armijo` but tests the
    *next direction* as well as the next point, so the search cannot be
    expressed as a :class:`~mopt.nonlinear.LineSearch` — it rebuilds
    :math:`d_{k+1}` for every trial step. The trial step starts from

    .. math::

        \\tau_k = \\frac{\\lvert g_k^T d_k \\rvert}{\\lVert d_k \\rVert^2},
        \\qquad \\Delta_k \\in [\\rho_1 \\tau_k, \\rho_2 \\tau_k],

    and shrinks by :math:`\\theta` until :math:`\\eta_k =
    \\theta^j \\Delta_k` makes both

    .. math::

        \\text{(i)} \\quad
            f(x_{k+1}) \\le f(x_k) + \\gamma \\eta_k g_k^T d_k,
        \\qquad
        \\text{(ii)} \\quad
            g_{k+1}^T d_{k+1} < -\\delta \\lVert g_{k+1} \\rVert^2 < 0

    hold. Condition (ii) *enforces* sufficient descent at every iteration,
    which is what makes :func:`polak_ribiere` safe here without the PRP+
    truncation. It is always satisfiable for small enough
    :math:`\\eta`: as :math:`\\eta \\to 0`, :math:`\\beta \\to 0` and
    :math:`d_{k+1} \\to -g_k`, giving :math:`-\\lVert g \\rVert^2 <
    -\\delta \\lVert g \\rVert^2` since :math:`\\delta < 1`.

    Note that (ii) is strict, so it cannot hold once :math:`g_{k+1} = 0`;
    convergence is therefore tested inside the search, and a trial step
    that satisfies (i) and lands on a stationary point is accepted.

    Parameters
    ----------
    beta : BetaRule
        Coefficient rule invoked as ``beta(grad_next, grad, d)``. Defaults
        to :func:`polak_ribiere`, the pairing the algorithm is written for.
    rho_1, rho_2 : float
        Bracket for the initial trial step, ``0 < rho_1 < rho_2``. The
        midpoint of ``[rho_1 * tau, rho_2 * tau]`` is used; since the
        search only shrinks, ``rho_2`` bounds the longest reachable step.
    gamma : float
        Sufficient-decrease constant of condition (i), in (0, 1).
    delta : float
        Sufficient-descent constant of condition (ii), in (0, 1); larger
        values demand a steeper next direction and force shorter steps.
    theta : float
        Shrink factor per rejected trial, in (0, 1).
    tol : float
        Convergence threshold on the gradient norm
        :math:`\\lVert \\nabla f(x) \\rVert_2`.
    max_iter : int
        Maximum number of iterations.
    max_shrink : int
        Maximum number of shrinks per line search before giving up.

    Raises
    ------
    ValueError
        If the tunables violate their stated ranges.
    """

    def __init__(
        self,
        beta: BetaRule = polak_ribiere,
        rho_1: float = 0.5,
        rho_2: float = 2.0,
        gamma: float = 0.14,
        delta: float = 0.87,
        theta: float = 0.5,
        tol: float = 1e-6,
        max_iter: int = 1000,
        max_shrink: int = 100,
    ):
        if not 0 < rho_1 < rho_2:
            raise ValueError("rho_1 and rho_2 must satisfy 0 < rho_1 < rho_2.")
        if not 0 < gamma < 1:
            raise ValueError("gamma must lie in (0, 1).")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1).")
        if not 0 < theta < 1:
            raise ValueError("theta must lie in (0, 1).")
        self.beta = beta
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        self.tol = tol
        self.max_iter = max_iter
        self.max_shrink = max_shrink

    def solve(self, problem: NLPProblem) -> OptimizeResult:
        x = problem.x0.copy()
        g = problem.gradient(x)
        d = -g
        for k in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message="Converged: gradient norm below tol.", n_iter=k,
                )
            fx = float(problem.f(x))
            slope = float(g @ d)
            if slope >= 0:
                return OptimizeResult(
                    x=x, fun=fx, success=False,
                    message=f"Direction is not a descent direction at "
                            f"iteration {k}.", n_iter=k,
                )

            # trial step tau, placed in [rho_1 tau, rho_2 tau]
            tau = abs(slope) / float(d @ d)
            eta = (self.rho_1 + self.rho_2) / 2 * tau

            accepted = False
            for _ in range(self.max_shrink):
                x_next = x + eta * d
                g_next = problem.gradient(x_next)
                gg_next = float(g_next @ g_next)
                d_next = -g_next + self.beta(g_next, g, d) * d
                if problem.f(x_next) <= fx + self.gamma * eta * slope:  # (i)
                    # (ii) is strict, so it cannot hold at a stationary
                    # point; accept the step and let the outer loop stop
                    if np.sqrt(gg_next) < self.tol:
                        accepted = True
                        break
                    if float(g_next @ d_next) < -self.delta * gg_next:  # (ii)
                        accepted = True
                        break
                eta *= self.theta
            if not accepted:
                return OptimizeResult(
                    x=x, fun=fx, success=False,
                    message=f"AM line search: no step meeting conditions (i) "
                            f"and (ii) within {self.max_shrink} shrinks.",
                    n_iter=k,
                )
            x, g, d = x_next, g_next, d_next
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message="Iteration limit reached.", n_iter=self.max_iter,
        )
