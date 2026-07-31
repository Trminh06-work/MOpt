"""First-order optimality conditions for constrained problems.

Two families live here, because a feasible set can be described two ways
and each supports a different test.

*Oracle form.* When the set is a projection or a linear minimizer, first
order optimality at :math:`x^\\star` is the variational inequality
:math:`\\nabla f(x^\\star)^T (y - x^\\star) \\ge 0` for every feasible
``y``. :func:`projected_gradient_residual` and :func:`frank_wolfe_gap`
each measure a violation of it, need no algebraic constraints, and are
exactly the quantities the matching solvers already drive to zero.

*Algebraic form.* When the constraints are written as
:math:`g(x) \\le 0` and :math:`h(x) = 0`, the same geometry becomes the
KKT conditions, with a multiplier per constraint —
:func:`lagrangian` and :func:`kkt_residual`. This form says *which*
constraints hold the solution in place and how hard, which the oracle form
cannot. It is also the form that needs a constraint qualification: KKT is
necessary at a local minimizer only when one holds (LICQ, MFCQ, Slater);
without one a minimizer can fail KKT, and the weaker Fritz John conditions
are what survive. See Nocedal & Wright, chapter 12.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mopt.nonlinear.problem import ConstrainedNLPProblem


def projected_gradient_residual(
    problem: ConstrainedNLPProblem,
    x: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Stationarity residual :math:`\\lVert P_S(x - \\alpha \\nabla f(x)) - x \\rVert`.

    Zero exactly at a first-order stationary point of the constrained
    problem, for any ``alpha > 0``, and positive otherwise. This is the
    quantity :class:`~mopt.nonlinear.ProjectedGradient` tests to decide it
    has converged, exposed here so any point — however it was produced —
    can be checked the same way.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Must carry a ``projection`` oracle.
    x : np.ndarray, shape (n,)
        Point to test; need not be feasible.
    alpha : float
        Step used to form the trial point. Any positive value has the same
        zero set, but the magnitude of the residual scales with it.

    Returns
    -------
    float
        The residual norm.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    x = np.asarray(x, dtype=float)
    step = problem.project(x - alpha * problem.gradient(x))
    return float(np.linalg.norm(step - x))


def frank_wolfe_gap(problem: ConstrainedNLPProblem, x: np.ndarray) -> float:
    """Frank-Wolfe gap :math:`\\max_{y \\in S} \\nabla f(x)^T (x - y)`.

    Nonnegative, zero exactly at a first-order stationary point, and an
    upper bound on :math:`f(x) - f^\\star` when ``f`` is convex — which
    makes it the one cheaply computable *certificate* of near-optimality
    among these residuals.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Must carry an ``lmo`` oracle.
    x : np.ndarray, shape (n,)
        Point to test.

    Returns
    -------
    float
        The gap.
    """
    if problem.lmo is None:
        raise ValueError("This problem has no lmo oracle.")
    x = np.asarray(x, dtype=float)
    g = problem.gradient(x)
    return float(g @ (x - np.asarray(problem.lmo(g), dtype=float)))


def lagrangian(
    problem: ConstrainedNLPProblem,
    x: np.ndarray,
    lam: np.ndarray | None = None,
    nu: np.ndarray | None = None,
) -> float:
    """Lagrangian :math:`L(x, \\lambda, \\nu) = f(x) + \\lambda^T g(x) + \\nu^T h(x)`.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Supplies ``f`` and the algebraic constraints.
    x : np.ndarray, shape (n,)
        Primal point.
    lam : np.ndarray, shape (m,), optional
        Inequality multipliers, required nonnegative by KKT but not
        checked here. Defaults to zeros.
    nu : np.ndarray, shape (p,), optional
        Equality multipliers, free in sign. Defaults to zeros.

    Returns
    -------
    float
        The Lagrangian value.
    """
    x = np.asarray(x, dtype=float)
    total = float(problem.f(x))
    if problem.ineq is not None:
        g = np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))
        lam = np.zeros(g.size) if lam is None else np.atleast_1d(lam)
        total += float(np.asarray(lam, dtype=float) @ g)
    if problem.eq is not None:
        h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
        nu = np.zeros(h.size) if nu is None else np.atleast_1d(nu)
        total += float(np.asarray(nu, dtype=float) @ h)
    return total


@dataclass
class KKTResidual:
    """How badly a point misses the KKT conditions, one number per condition.

    All four are nonnegative and zero exactly when the corresponding
    condition holds, so :attr:`max_violation` is a single summary.

    Attributes
    ----------
    stationarity : float
        :math:`\\lVert \\nabla f + J_g^T \\lambda + J_h^T \\nu \\rVert`.
    primal_feasibility : float
        Worst constraint violation, ``max(max(g, 0), |h|)``.
    dual_feasibility : float
        :math:`\\max(0, -\\min_i \\lambda_i)`; nonzero when an inequality
        multiplier came out negative, which means the constraint is
        pushing the wrong way.
    complementarity : float
        :math:`\\max_i \\lvert \\lambda_i g_i(x) \\rvert`; nonzero when an
        inactive constraint carries a multiplier.
    lam : np.ndarray
        Inequality multipliers, supplied or estimated.
    nu : np.ndarray
        Equality multipliers, supplied or estimated.
    active : np.ndarray
        Boolean mask of inequalities treated as active.
    """

    stationarity: float
    primal_feasibility: float
    dual_feasibility: float
    complementarity: float
    lam: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def max_violation(self) -> float:
        """The largest of the four residuals."""
        return max(
            self.stationarity,
            self.primal_feasibility,
            self.dual_feasibility,
            self.complementarity,
        )

    def satisfied(self, tol: float = 1e-6) -> bool:
        """Whether every condition holds to within ``tol``."""
        return self.max_violation <= tol


def kkt_residual(
    problem: ConstrainedNLPProblem,
    x: np.ndarray,
    lam: np.ndarray | None = None,
    nu: np.ndarray | None = None,
    active_tol: float = 1e-8,
) -> KKTResidual:
    """Measure the KKT conditions at ``x``, estimating multipliers if needed.

    The conditions, for :math:`\\min f(x)` subject to :math:`g(x) \\le 0`
    and :math:`h(x) = 0`, are

    .. math::

        \\nabla f + J_g^T \\lambda + J_h^T \\nu = 0, \\quad
        g \\le 0, \\; h = 0, \\quad
        \\lambda \\ge 0, \\quad
        \\lambda_i g_i = 0.

    When ``lam`` and ``nu`` are not given they are recovered from the
    stationarity equation by least squares, over the *active* inequalities
    only — inactive ones must carry a zero multiplier by complementary
    slackness, so including them would let the fit hide a stationarity
    violation. The fit is unconstrained, deliberately: forcing
    :math:`\\lambda \\ge 0` would mask exactly the failure that
    ``dual_feasibility`` is there to report.

    A large ``stationarity`` residual with no obvious cause usually means
    the active constraint gradients are linearly dependent, i.e. LICQ
    fails; the multipliers are then not unique and least squares returns
    the least-norm choice.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Must carry ``ineq`` and/or ``eq``; an oracle alone is not enough.
    x : np.ndarray, shape (n,)
        Point to test.
    lam, nu : np.ndarray, optional
        Known multipliers. When omitted, both are estimated.
    active_tol : float
        An inequality counts as active when ``g_i(x) >= -active_tol``.

    Returns
    -------
    KKTResidual
        The four residuals and the multipliers used.

    Raises
    ------
    ValueError
        If the problem carries no algebraic constraints.
    """
    if problem.ineq is None and problem.eq is None:
        raise ValueError(
            "kkt_residual needs algebraic constraints: set problem.ineq "
            "and/or problem.eq. A projection or lmo oracle alone does not "
            "identify individual constraints, so it carries no multipliers."
        )
    x = np.asarray(x, dtype=float)
    grad_f = problem.gradient(x)

    g = (np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))
         if problem.ineq is not None else np.zeros(0))
    h = (np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
         if problem.eq is not None else np.zeros(0))
    Jg = problem.ineq_jacobian(x)
    Jh = problem.eq_jacobian(x)
    active = g >= -active_tol

    if lam is None or nu is None:
        # solve  [Jg_active; Jh]^T [lam_active; nu] = -grad_f  in least squares
        rows = np.vstack([Jg[active], Jh]) if Jg.size or Jh.size else np.zeros((0, x.size))
        if rows.shape[0] > 0:
            solution, *_ = np.linalg.lstsq(rows.T, -grad_f, rcond=None)
        else:
            solution = np.zeros(0)
        n_active = int(active.sum())
        if lam is None:
            lam = np.zeros(g.size)
            lam[active] = solution[:n_active]
        if nu is None:
            nu = solution[n_active:]

    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    nu = np.atleast_1d(np.asarray(nu, dtype=float)) if h.size else np.zeros(0)

    stationarity = grad_f.copy()
    if lam.size:
        stationarity = stationarity + Jg.T @ lam
    if nu.size:
        stationarity = stationarity + Jh.T @ nu

    primal = 0.0
    if g.size:
        primal = max(primal, float(np.max(g)))
    if h.size:
        primal = max(primal, float(np.max(np.abs(h))))

    return KKTResidual(
        stationarity=float(np.linalg.norm(stationarity)),
        primal_feasibility=max(0.0, primal),
        dual_feasibility=float(max(0.0, -np.min(lam))) if lam.size else 0.0,
        complementarity=float(np.max(np.abs(lam * g))) if g.size else 0.0,
        lam=lam,
        nu=nu,
        active=active,
    )
