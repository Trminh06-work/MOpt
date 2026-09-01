"""Sequential quadratic programming.

SQP treats a constrained problem as a sequence of quadratic programs. At
each iterate it builds a quadratic model of the Lagrangian subject to the
*linearized* constraints, solves that for a step, and moves along it as
far as a merit function allows:

.. math::

    \\min_d \\; \\nabla f(x_k)^T d + \\tfrac{1}{2} d^T B_k d
    \\quad \\text{s.t.} \\quad
    g(x_k) + J_g d \\le 0, \\;\\; h(x_k) + J_h d = 0.

Unlike the sequential methods in
:mod:`~mopt.nonlinear.constrained.penalty`, the constraints are never
approximated away — each subproblem carries them explicitly — which is
what gives SQP its fast local convergence. It is the method underlying
``scipy.optimize.minimize(method="SLSQP")``.

Three pieces make it work away from the solution, and each is a separate
concern below: :func:`damped_bfgs` keeps the model matrix positive
definite even where the Lagrangian is not convex, an L1 exact-penalty
merit function decides how far to step, and multipliers are re-estimated
from each subproblem to drive the KKT convergence test.
"""

from __future__ import annotations

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.constrained.optimality import kkt_residual
from mopt.nonlinear.constrained.penalty import PenaltyMethod
from mopt.nonlinear.problem import ConstrainedNLPProblem
from mopt.nonlinear.unconstrained.trust_region import TrustRegion, dogleg


def damped_bfgs(
    B: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    damping: float = 0.2,
) -> np.ndarray:
    """Powell-damped BFGS update of a direct Hessian approximation.

    The plain BFGS update keeps ``B`` positive definite only when the
    curvature condition :math:`s^T y > 0` holds. Minimizing a *Lagrangian*
    offers no such guarantee — it is a saddle function, not a convex one —
    so Powell's modification replaces ``y`` by the interpolant

    .. math::

        \\hat{y} = \\theta y + (1 - \\theta) B s, \\qquad
        \\theta = \\frac{(1 - \\rho)\\, s^T B s}{s^T B s - s^T y},

    chosen so that :math:`s^T \\hat{y} \\ge \\rho\\, s^T B s > 0`, and
    then applies the standard update

    .. math::

        B_{k+1} = B_k - \\frac{B_k s s^T B_k}{s^T B_k s}
                      + \\frac{\\hat{y} \\hat{y}^T}{\\hat{y}^T s}.

    When the curvature condition already holds comfortably
    (:math:`s^T y \\ge \\rho\\, s^T B s`) then :math:`\\theta = 1`,
    :math:`\\hat{y} = y`, and this is exactly BFGS.

    Note this updates the Hessian approximation ``B`` directly, unlike
    :func:`~mopt.nonlinear.bfgs`, which updates the *inverse* ``H``: SQP
    needs ``B`` itself, because it appears as the quadratic term of the
    subproblem rather than as a multiplier on the gradient.

    Parameters
    ----------
    B : np.ndarray, shape (n, n)
        Current positive definite approximation.
    s : np.ndarray, shape (n,)
        Step taken, ``x_next - x``.
    y : np.ndarray, shape (n,)
        Change in the gradient of the Lagrangian across that step.
    damping : float
        The :math:`\\rho` above, in ``(0, 1)``; Powell's choice is 0.2.
        Larger values damp more aggressively, keeping ``B`` better
        conditioned at the cost of tracking true curvature less closely.

    Returns
    -------
    np.ndarray, shape (n, n)
        The updated approximation, positive definite whenever ``B`` was.

    Raises
    ------
    ValueError
        If ``damping`` is out of range, or ``s @ B @ s <= 0`` (which means
        ``B`` was not positive definite, or ``s`` is zero).
    """
    if not 0 < damping < 1:
        raise ValueError("damping must lie in (0, 1).")
    B = np.asarray(B, dtype=float)
    Bs = B @ s
    sBs = float(s @ Bs)
    if sBs <= 0:
        raise ValueError(
            "s @ B @ s <= 0: B is not positive definite, or the step is zero."
        )
    sy = float(s @ y)
    if sy >= damping * sBs:
        y_hat = np.asarray(y, dtype=float)
    else:
        theta = (1.0 - damping) * sBs / (sBs - sy)
        y_hat = theta * y + (1.0 - theta) * Bs
    return B - np.outer(Bs, Bs) / sBs + np.outer(y_hat, y_hat) / float(y_hat @ s)


def _constraints(problem: ConstrainedNLPProblem, x: np.ndarray):
    """Return ``(g, h, J_g, J_h)`` at ``x``, empty where absent."""
    n = x.size
    if problem.ineq is not None:
        g = np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))
        Jg = problem.ineq_jacobian(x)
    else:
        g, Jg = np.zeros(0), np.zeros((0, n))
    if problem.eq is not None:
        h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
        Jh = problem.eq_jacobian(x)
    else:
        h, Jh = np.zeros(0), np.zeros((0, n))
    return g, h, Jg, Jh


def _grad_lagrangian(grad_f, Jg, Jh, lam, nu):
    """:math:`\\nabla f + J_g^T \\lambda + J_h^T \\nu`."""
    return grad_f + Jg.T @ lam + Jh.T @ nu


def _qp_multipliers(residual, g_linear, Jg, Jh, active_tol):
    """Least-squares multipliers for the solved subproblem.

    Fits ``residual + J_g[active]^T lam + J_h^T nu = 0`` over the active
    linearized inequalities only — inactive ones must carry a zero
    multiplier by complementary slackness — then clips ``lam`` at zero to
    respect dual feasibility.
    """
    m, p = Jg.shape[0], Jh.shape[0]
    active = g_linear >= -active_tol if m else np.zeros(0, dtype=bool)
    blocks = [J for J in (Jg[active], Jh) if J.size]
    lam = np.zeros(m)
    nu = np.zeros(p)
    if not blocks:
        return lam, nu
    A = np.vstack(blocks)
    z = np.linalg.lstsq(A.T, -residual, rcond=None)[0]
    n_active = int(active.sum())
    if n_active:
        lam[active] = np.maximum(z[:n_active], 0.0)
    if p:
        nu = z[n_active:]
    return lam, nu


class SQP(BaseOptimizer):
    """Sequential quadratic programming with a damped BFGS model.

    Each iteration solves the quadratic subproblem in the module
    docstring for a step ``d``, estimates fresh multipliers from it, and
    line searches along ``d`` on the L1 exact-penalty merit function

    .. math::

        P(x; \\mu) = f(x) + \\mu \\left(
            \\sum_i \\max(g_i(x), 0) + \\sum_j |h_j(x)| \\right).

    That merit function is *exact*: for ``mu`` above the largest
    multiplier, its minimizers coincide with the constrained ones, which
    is why the weight is refreshed as
    ``mu <- max(mu, 1.1 * max|multiplier|)`` each iteration. Its
    directional derivative along a subproblem solution is
    :math:`\\nabla f^T d - \\mu (\\lVert g^{+}\\rVert_1 + \\lVert h
    \\rVert_1)`, which the Armijo test uses.

    The model matrix starts at the identity and is refreshed by
    :func:`damped_bfgs` on the Lagrangian gradient, so no Hessian is
    needed — ``problem.hess`` is ignored entirely.

    Convergence is measured by
    :func:`~mopt.nonlinear.kkt_residual` at the current multipliers, so a
    successful run has certified stationarity, primal and dual
    feasibility, and complementarity together.

    Requires ``problem.ineq`` and/or ``problem.eq``. ``problem.x0`` need
    not be feasible.

    Parameters
    ----------
    qp_solver : BaseOptimizer, optional
        Solver for the quadratic subproblem, which is handed over as a
        :class:`~mopt.nonlinear.ConstrainedNLPProblem` in the step
        variable. Defaults to a deliberately hard-driven
        :class:`~mopt.nonlinear.PenaltyMethod` — see the Notes on why its
        ``tau_max`` is so large.
    tol : float
        Convergence threshold on the KKT residual.
    max_iter : int
        Maximum number of outer iterations.
    gamma : float
        Armijo constant for the merit line search, in ``(0, 1/2)``.
    delta : float
        Backtracking shrink factor for the merit line search, in
        ``(0, 1)``.
    max_shrink : int
        Maximum backtracking steps per iteration.
    damping : float
        Powell damping parameter passed to :func:`damped_bfgs`.
    active_tol : float
        A linearized inequality counts as active when
        ``g_i + J_g[i] @ d >= -active_tol``, which decides whether it
        carries a multiplier.

    Notes
    -----
    The subproblem is solved by penalizing the linearized constraints
    rather than by an active-set or interior-point QP routine, which
    keeps the whole method built from pieces already in this package.
    The cost is that the step satisfies those constraints only to within
    the penalty method's own error, roughly ``|multiplier| / tau_max``,
    and that residual is what eventually stops progress here: once a
    step buys less merit decrease than the residual violation costs, the
    line search rejects it and the run stalls. It stalls *near* the
    solution — the iterates converge superlinearly right up to that
    point — so the symptom is a run that hits ``max_iter`` with an
    accurate ``fun`` and a KKT residual just above ``tol``.

    That is why the default ``tau_max`` is ``1e13`` rather than the far
    gentler value :class:`~mopt.nonlinear.PenaltyMethod` uses on its own.
    The subproblem tolerates it because its merit function is *exactly*
    quadratic — the constraints are linear in the step — so the inner
    trust region solves it in a few iterations without ever needing the
    ill-conditioned directions to be accurate. Lowering it to ``1e9``
    visibly costs accuracy; raising it to ``1e15`` buys none.

    A step that is not a descent direction for the merit function — from
    inconsistent linearized constraints, or a subproblem that stopped
    early — is met by raising ``mu``, and reported as a failure if that
    does not recover it.
    """

    def __init__(
        self,
        qp_solver: BaseOptimizer | None = None,
        tol: float = 1e-6,
        max_iter: int = 50,
        gamma: float = 0.1,
        delta: float = 0.5,
        max_shrink: int = 60,
        damping: float = 0.2,
        active_tol: float = 1e-8,
    ):
        if not 0 < gamma < 0.5:
            raise ValueError("gamma must lie in (0, 1/2).")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1).")
        self.qp_solver = qp_solver
        self.tol = tol
        self.max_iter = max_iter
        self.gamma = gamma
        self.delta = delta
        self.max_shrink = max_shrink
        self.damping = damping
        self.active_tol = active_tol

    def solve(self, problem: ConstrainedNLPProblem) -> OptimizeResult:
        if getattr(problem, "ineq", None) is None and (
            getattr(problem, "eq", None) is None
        ):
            raise ValueError(
                "SQP requires algebraic constraints: set problem.ineq "
                "and/or problem.eq."
            )
        qp_solver = self.qp_solver or PenaltyMethod(
            tau_max=1e13,
            tol=1e-12,
            inner_solver=TrustRegion(method=dogleg, tol=1e-13, radius_tol=1e-15),
        )
        x = np.asarray(problem.x0, dtype=float).copy()
        n = x.size
        B = np.eye(n)
        d = np.zeros(n)
        mu = 1.0

        g, h, Jg, Jh = _constraints(problem, x)
        lam, nu = np.zeros(g.size), np.zeros(h.size)

        for k in range(self.max_iter):
            grad_f = np.asarray(problem.gradient(x), dtype=float)

            residual = kkt_residual(problem, x, lam, nu)
            if residual.max_violation < self.tol:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message=f"Converged: KKT residual "
                            f"{residual.max_violation:.3g} below tol.",
                    n_iter=k,
                )

            d, message = self._qp_step(
                grad_f, B, g, h, Jg, Jh, d, qp_solver
            )
            if d is None:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=message, n_iter=k,
                )

            lam_hat, nu_hat = _qp_multipliers(
                grad_f + B @ d, g + Jg @ d, Jg, Jh, self.active_tol
            )

            # merit weight must dominate the multipliers for exactness
            largest = max(
                float(np.abs(lam_hat).max()) if lam_hat.size else 0.0,
                float(np.abs(nu_hat).max()) if nu_hat.size else 0.0,
            )
            mu = max(mu, 1.1 * largest)

            violation = float(np.maximum(g, 0.0).sum() + np.abs(h).sum())
            slope = float(grad_f @ d) - mu * violation
            while slope >= 0 and violation > 0 and mu < 1e12:
                mu *= 2.0  # more weight on the constraints tilts it negative
                slope = float(grad_f @ d) - mu * violation
            if slope >= 0:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Subproblem step is not a descent direction for "
                            f"the merit function at iteration {k}; the "
                            "linearized constraints are likely inconsistent.",
                    n_iter=k,
                )

            alpha = self._merit_line_search(problem, x, d, mu, slope)
            x_next = x + alpha * d

            g_next, h_next, Jg_next, Jh_next = _constraints(problem, x_next)
            s = alpha * d
            if float(s @ s) > 0:
                y = _grad_lagrangian(
                    np.asarray(problem.gradient(x_next), dtype=float),
                    Jg_next, Jh_next, lam_hat, nu_hat,
                ) - _grad_lagrangian(grad_f, Jg, Jh, lam_hat, nu_hat)
                try:
                    B = damped_bfgs(B, s, y, self.damping)
                except ValueError:
                    B = np.eye(n)  # reset rather than carry a broken model

            x, lam, nu = x_next, lam_hat, nu_hat
            g, h, Jg, Jh = g_next, h_next, Jg_next, Jh_next

        residual = kkt_residual(problem, x, lam, nu)
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message=f"Iteration limit reached with KKT residual "
                    f"{residual.max_violation:.3g}.",
            n_iter=self.max_iter,
        )

    def _qp_step(self, grad_f, B, g, h, Jg, Jh, d0, qp_solver):
        """Solve the linearized subproblem for a step; ``(d, message)``."""
        subproblem = ConstrainedNLPProblem(
            f=lambda step: float(grad_f @ step + 0.5 * step @ B @ step),
            x0=d0,
            grad=lambda step: grad_f + B @ step,
            hess=lambda step: B,
            ineq=(lambda step: g + Jg @ step) if g.size else None,
            ineq_jac=(lambda step: Jg) if g.size else None,
            eq=(lambda step: h + Jh @ step) if h.size else None,
            eq_jac=(lambda step: Jh) if h.size else None,
        )
        try:
            result = qp_solver.solve(subproblem)
        except ValueError as exc:
            return None, f"Quadratic subproblem failed: {exc}"
        if result.x is None:
            return None, f"Quadratic subproblem returned no step: {result.message}"
        return np.asarray(result.x, dtype=float), ""

    def _merit_line_search(self, problem, x, d, mu, slope) -> float:
        """Backtrack on the L1 merit function; always returns a step."""

        def merit(z: np.ndarray) -> float:
            g, h, _, _ = _constraints(problem, np.asarray(z, dtype=float))
            return float(problem.f(z)) + mu * float(
                np.maximum(g, 0.0).sum() + np.abs(h).sum()
            )

        base = merit(x)
        alpha = 1.0  # the unit step first, for fast local convergence
        for _ in range(self.max_shrink):
            if merit(x + alpha * d) <= base + self.gamma * alpha * slope:
                return alpha
            alpha *= self.delta
        return alpha
