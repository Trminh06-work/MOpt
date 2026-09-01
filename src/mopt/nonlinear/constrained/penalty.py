"""Penalty and barrier methods for constrained nonlinear programs.

Both solve a constrained problem as a *sequence* of unconstrained ones:
the constraints are folded into the objective through a merit function
governed by a scalar ``tau``, which is driven toward its limit one outer
iteration at a time, each warm-started from the last.

They differ in how the constraints enter, and therefore in where the
iterates live. The quadratic penalty charges for violation, so iterates
approach the feasible set from *outside* and only become feasible in the
limit ``tau -> infinity``. The logarithmic barrier charges for proximity
to the boundary, which is infinite outside it, so iterates stay strictly
*inside* and approach the boundary as ``tau -> 0``; that requires a
strictly feasible starting point, and cannot handle equality constraints
(which have no interior) — those keep a penalty term.

The merit functions are built from the problem's algebraic ``ineq`` and
``eq``, so unlike :mod:`~mopt.nonlinear.constrained.projected_gradient`
an oracle is not enough here.
"""

from __future__ import annotations

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.nonlinear.problem import ConstrainedNLPProblem, NLPProblem
from mopt.nonlinear.unconstrained.quasi_newton import QuasiNewton
from mopt.nonlinear.unconstrained.trust_region import TrustRegion, dogleg


def constraint_violation(problem: ConstrainedNLPProblem, x: np.ndarray) -> float:
    """Worst constraint violation at ``x``.

    Returns :math:`\\max(\\max_i g_i(x), \\max_j |h_j(x)|, 0)`, which is
    zero exactly at a feasible point and is what the sequential methods
    test for convergence.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Must carry ``ineq`` and/or ``eq``.
    x : np.ndarray, shape (n,)
        Point to measure.

    Returns
    -------
    float
        The violation, never negative.
    """
    x = np.asarray(x, dtype=float)
    worst = 0.0
    if problem.ineq is not None:
        g = np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))
        if g.size:
            worst = max(worst, float(g.max()))
    if problem.eq is not None:
        h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
        if h.size:
            worst = max(worst, float(np.abs(h).max()))
    return worst


def penalty_merit(
    problem: ConstrainedNLPProblem,
    tau: float,
    x0: np.ndarray | None = None,
) -> NLPProblem:
    """Quadratic-penalty merit function as an unconstrained problem.

    .. math::

        P(x; \\tau) = f(x) + \\tau \\left(
            \\sum_i \\max(g_i(x), 0)^2 + \\sum_j h_j(x)^2 \\right)

    with gradient
    :math:`\\nabla f + 2\\tau (J_g^T g^{+} + J_h^T h)`, where
    :math:`g^{+} = \\max(g, 0)`. The returned problem is an ordinary
    :class:`~mopt.nonlinear.NLPProblem`, so any unconstrained solver can
    minimize it.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Must carry ``ineq`` and/or ``eq``.
    tau : float
        Penalty weight, positive. Larger values enforce the constraints
        harder and condition the merit function worse.
    x0 : np.ndarray, optional
        Starting point for the returned problem; defaults to
        ``problem.x0``. The sequential methods pass the previous outer
        iterate, which warm-starts the inner solve.

    Returns
    -------
    NLPProblem
        The merit function, its gradient, and — when ``problem.hess`` is
        available — its Hessian.

    Notes
    -----
    The Hessian is the **Gauss-Newton** approximation

    .. math::

        \\nabla^2 P \\approx \\nabla^2 f
            + 2\\tau \\left( J_{g,\\mathrm{active}}^T J_{g,\\mathrm{active}}
                           + J_h^T J_h \\right),

    dropping the :math:`2\\tau \\sum_i g_i^{+} \\nabla^2 g_i` and
    :math:`2\\tau \\sum_j h_j \\nabla^2 h_j` terms, since
    :class:`ConstrainedNLPProblem` carries constraint Jacobians but not
    constraint Hessians. It is exact for affine constraints, and the
    dropped terms vanish as the iterates approach feasibility. The
    approximation is positive semidefinite by construction, so it never
    turns a convex merit function indefinite.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    x0 = problem.x0 if x0 is None else x0

    def merit(x: np.ndarray) -> float:
        total = float(problem.f(x))
        if problem.ineq is not None:
            violation = np.maximum(
                np.atleast_1d(np.asarray(problem.ineq(x), dtype=float)), 0.0
            )
            total += tau * float(violation @ violation)
        if problem.eq is not None:
            h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
            total += tau * float(h @ h)
        return total

    def merit_grad(x: np.ndarray) -> np.ndarray:
        out = np.asarray(problem.gradient(x), dtype=float)
        if problem.ineq is not None:
            violation = np.maximum(
                np.atleast_1d(np.asarray(problem.ineq(x), dtype=float)), 0.0
            )
            out = out + 2.0 * tau * (problem.ineq_jacobian(x).T @ violation)
        if problem.eq is not None:
            h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
            out = out + 2.0 * tau * (problem.eq_jacobian(x).T @ h)
        return out

    def merit_hess(x: np.ndarray) -> np.ndarray:
        H = np.asarray(problem.hess(x), dtype=float)
        if problem.ineq is not None:
            g = np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))
            active = g > 0.0  # inactive constraints contribute nothing
            if active.any():
                J = problem.ineq_jacobian(x)[active]
                H = H + 2.0 * tau * (J.T @ J)
        if problem.eq is not None:
            J = problem.eq_jacobian(x)
            if J.size:
                H = H + 2.0 * tau * (J.T @ J)
        return H

    return NLPProblem(
        f=merit,
        x0=x0,
        grad=merit_grad,
        hess=None if problem.hess is None else merit_hess,
    )


def barrier_merit(
    problem: ConstrainedNLPProblem,
    tau: float,
    x0: np.ndarray | None = None,
) -> NLPProblem:
    """Logarithmic-barrier merit function as an unconstrained problem.

    .. math::

        B(x; \\tau) = f(x)
            - \\tau \\sum_i \\log(-g_i(x))
            + \\frac{1}{\\tau} \\sum_j h_j(x)^2

    Inequalities become a barrier that is ``+inf`` outside the feasible
    set, keeping every iterate strictly inside; equalities keep a penalty
    term, whose weight :math:`1/\\tau` grows as the barrier weight
    :math:`\\tau` shrinks. The gradient is
    :math:`\\nabla f + \\tau J_g^T (1/s) + (2/\\tau) J_h^T h` with
    :math:`s = -g > 0`.

    Parameters
    ----------
    problem : ConstrainedNLPProblem
        Must carry ``ineq``; ``eq`` is optional.
    tau : float
        Barrier weight, positive. Smaller values let iterates approach
        the boundary and condition the merit function worse.
    x0 : np.ndarray, optional
        Starting point for the returned problem; defaults to
        ``problem.x0``. Must be strictly feasible, or the merit function
        is ``+inf`` there.

    Returns
    -------
    NLPProblem
        The merit function, its gradient, and — when ``problem.hess`` is
        available — its Hessian.

    Raises
    ------
    ValueError
        If ``tau`` is not positive, or the problem has no inequality
        constraints (there is nothing to put a barrier on).

    Notes
    -----
    As with :func:`penalty_merit` the Hessian drops the constraint
    curvature terms, keeping
    :math:`\\tau \\sum_i \\nabla g_i \\nabla g_i^T / s_i^2
    + (2/\\tau) J_h^T J_h`.

    The merit function evaluates to ``+inf`` at infeasible points, which
    is enough for descent methods that only compare objective values
    there (backtracking line searches, trust-region ratio tests) — the
    step is simply rejected. Its *gradient* outside the feasible set is
    finite but meaningless, so prefer an inner solver that evaluates
    gradients only at accepted iterates.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if problem.ineq is None:
        raise ValueError(
            "barrier_merit needs inequality constraints: set problem.ineq. "
            "For equalities alone use penalty_merit."
        )
    x0 = problem.x0 if x0 is None else x0

    def slack(x: np.ndarray) -> np.ndarray:
        return -np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))

    def merit(x: np.ndarray) -> float:
        s = slack(x)
        if np.any(s <= 0.0):
            return np.inf  # outside the feasible set
        total = float(problem.f(x)) - tau * float(np.log(s).sum())
        if problem.eq is not None:
            h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
            total += float(h @ h) / tau
        return total

    def merit_grad(x: np.ndarray) -> np.ndarray:
        s = slack(x)
        out = np.asarray(problem.gradient(x), dtype=float)
        out = out + tau * (problem.ineq_jacobian(x).T @ (1.0 / s))
        if problem.eq is not None:
            h = np.atleast_1d(np.asarray(problem.eq(x), dtype=float))
            out = out + (2.0 / tau) * (problem.eq_jacobian(x).T @ h)
        return out

    def merit_hess(x: np.ndarray) -> np.ndarray:
        s = slack(x)
        H = np.asarray(problem.hess(x), dtype=float)
        scaled = problem.ineq_jacobian(x) / s[:, None]
        H = H + tau * (scaled.T @ scaled)
        if problem.eq is not None:
            J = problem.eq_jacobian(x)
            if J.size:
                H = H + (2.0 / tau) * (J.T @ J)
        return H

    return NLPProblem(
        f=merit,
        x0=x0,
        grad=merit_grad,
        hess=None if problem.hess is None else merit_hess,
    )


def _default_inner(problem: NLPProblem) -> BaseOptimizer:
    """Pick an inner solver: dogleg trust region if a Hessian is available."""
    if problem.hess is not None:
        return TrustRegion(method=dogleg)
    return QuasiNewton()


class PenaltyMethod(BaseOptimizer):
    """Sequential quadratic-penalty method.

    Minimizes :func:`penalty_merit` at an increasing sequence of penalty
    weights ``tau <- tau / theta``, each solve warm-started from the last
    iterate, until the constraint violation falls below ``tol`` or
    ``tau`` reaches ``tau_max``.

    Iterates are generally *infeasible* — the penalty only charges for
    violation, it does not forbid it — and become feasible in the limit.
    That makes the method indifferent to where it starts, unlike
    :class:`PenaltyBarrierMethod`. The price is conditioning: the merit
    Hessian carries ``tau`` in the constraint directions, so it grows
    steadily worse, which is why ``tau_max`` exists and why the outer
    loop escalates gradually rather than jumping straight to a huge
    weight.

    Requires ``problem.ineq`` and/or ``problem.eq``.

    Parameters
    ----------
    inner_solver : BaseOptimizer, optional
        Solver for each unconstrained merit problem. Defaults to
        :class:`~mopt.nonlinear.TrustRegion` with
        :func:`~mopt.nonlinear.dogleg` when ``problem.hess`` is
        available, otherwise :class:`~mopt.nonlinear.QuasiNewton`. Trust
        region handles the ill-conditioning of large ``tau`` better than
        line-search methods do.
    tau0 : float
        Initial penalty weight.
    theta : float
        Escalation factor in ``(0, 1)``; ``tau`` is divided by it each
        outer iteration, so smaller values escalate faster.
    tau_max : float
        Largest weight to use, bounding the conditioning damage.
    tol : float
        Convergence threshold on :func:`constraint_violation`.
    max_iter : int
        Maximum number of outer iterations.

    Notes
    -----
    ``n_iter`` on the result counts *outer* iterations; the total inner
    iteration count is reported in ``message``.
    """

    def __init__(
        self,
        inner_solver: BaseOptimizer | None = None,
        tau0: float = 1.0,
        theta: float = 0.1,
        tau_max: float = 1e7,
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        if tau0 <= 0:
            raise ValueError("tau0 must be positive.")
        if not 0 < theta < 1:
            raise ValueError("theta must lie in (0, 1).")
        self.inner_solver = inner_solver
        self.tau0 = tau0
        self.theta = theta
        self.tau_max = tau_max
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: ConstrainedNLPProblem) -> OptimizeResult:
        if getattr(problem, "ineq", None) is None and (
            getattr(problem, "eq", None) is None
        ):
            raise ValueError(
                "PenaltyMethod requires algebraic constraints: set "
                "problem.ineq and/or problem.eq."
            )
        x = np.asarray(problem.x0, dtype=float).copy()
        tau = self.tau0
        inner_total = 0
        inner_note = ""
        for k in range(self.max_iter):
            merit = penalty_merit(problem, tau, x)
            solver = self.inner_solver or _default_inner(merit)
            try:
                inner = solver.solve(merit)
            except ValueError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Inner solver failed at tau={tau:.3g}: {exc}",
                    n_iter=k,
                )
            inner_total += inner.n_iter
            if inner.x is not None:
                x = np.asarray(inner.x, dtype=float)
            violation = constraint_violation(problem, x)
            # feasibility alone is not convergence: an inner solve that gave
            # up may leave a feasible but far-from-optimal iterate
            if violation <= self.tol and inner.success:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=True,
                    message=f"Converged: constraint violation {violation:.3g} "
                            f"below tol ({inner_total} inner iterations).",
                    n_iter=k + 1,
                )
            if not inner.success:
                inner_note = f" Last inner solve did not converge: {inner.message}"
            if tau >= self.tau_max:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Penalty weight reached tau_max={self.tau_max:.3g} "
                            f"with violation {violation:.3g}.{inner_note}",
                    n_iter=k + 1,
                )
            tau = min(tau / self.theta, self.tau_max)
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message=f"Outer iteration limit reached with violation "
                    f"{constraint_violation(problem, x):.3g}.{inner_note}",
            n_iter=self.max_iter,
        )


class PenaltyBarrierMethod(BaseOptimizer):
    """Sequential penalty-barrier (interior point) method.

    Minimizes :func:`barrier_merit` at a *decreasing* sequence of weights
    ``tau <- tau * theta``, each solve warm-started from the last
    iterate, until ``tau`` reaches ``tau_min``.

    Every iterate is strictly feasible for the inequalities, so the run
    can be stopped at any point and still yield a usable answer — the
    complement of :class:`PenaltyMethod`, which only reaches feasibility
    at the end. Equality constraints have no interior, so they are
    handled by a penalty term of weight :math:`1/\\tau` instead, and
    those *are* only satisfied in the limit; success is therefore
    reported on the equality violation.

    Requires ``problem.ineq``, and a strictly feasible ``problem.x0``.

    Parameters
    ----------
    inner_solver : BaseOptimizer, optional
        Solver for each merit problem; see :class:`PenaltyMethod`. A
        trust-region default is the safer choice here: the merit function
        is ``+inf`` outside the feasible set, and the ratio test simply
        rejects such steps.
    tau0 : float
        Initial barrier weight.
    theta : float
        Reduction factor in ``(0, 1)``; ``tau`` is multiplied by it each
        outer iteration.
    tau_min : float
        Smallest weight to use, bounding the conditioning damage.
    tol : float
        Convergence threshold on :func:`constraint_violation`.
    max_iter : int
        Maximum number of outer iterations.

    Raises
    ------
    ValueError
        If the problem has no inequality constraints, or ``problem.x0``
        is not strictly feasible for them.

    Notes
    -----
    Success is *not* reported on feasibility alone. Every iterate here is
    feasible by construction, so the violation says nothing about
    optimality; a run only succeeds when the final inner minimization
    converged too, and the reason it did not is passed through in
    ``message``.

    The default dogleg inner solver needs a positive definite merit
    Hessian, which the barrier term supplies wherever the constraint
    gradients span the space — but not everywhere. Starting from a point
    where they do not (an interior point of a ball, say, where
    :math:`\\nabla g = 0`) with an objective that contributes no
    curvature of its own leaves the model singular and the subproblem
    fails. Pass ``TrustRegion(method=cauchy)``, which is content with a
    singular model, or start somewhere less symmetric.
    """

    def __init__(
        self,
        inner_solver: BaseOptimizer | None = None,
        tau0: float = 1.0,
        theta: float = 0.1,
        tau_min: float = 1e-7,
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        if tau0 <= 0:
            raise ValueError("tau0 must be positive.")
        if not 0 < theta < 1:
            raise ValueError("theta must lie in (0, 1).")
        self.inner_solver = inner_solver
        self.tau0 = tau0
        self.theta = theta
        self.tau_min = tau_min
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: ConstrainedNLPProblem) -> OptimizeResult:
        if getattr(problem, "ineq", None) is None:
            raise ValueError(
                "PenaltyBarrierMethod requires inequality constraints: set "
                "problem.ineq. For equalities alone use PenaltyMethod."
            )
        x = np.asarray(problem.x0, dtype=float).copy()
        g0 = np.atleast_1d(np.asarray(problem.ineq(x), dtype=float))
        if np.any(g0 >= 0):
            raise ValueError(
                "PenaltyBarrierMethod needs a strictly feasible x0: "
                f"max(ineq(x0)) = {float(g0.max()):.3g}, must be < 0."
            )
        tau = self.tau0
        inner_total = 0
        inner_ok = False
        inner_note = ""
        for k in range(self.max_iter):
            merit = barrier_merit(problem, tau, x)
            solver = self.inner_solver or _default_inner(merit)
            try:
                inner = solver.solve(merit)
            except ValueError as exc:
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=False,
                    message=f"Inner solver failed at tau={tau:.3g}: {exc}",
                    n_iter=k,
                )
            inner_total += inner.n_iter
            inner_ok = inner.success
            if not inner_ok:
                inner_note = f" Last inner solve did not converge: {inner.message}"
            if inner.x is not None and np.isfinite(merit.f(inner.x)):
                x = np.asarray(inner.x, dtype=float)  # never leave the interior
            violation = constraint_violation(problem, x)
            if tau <= self.tau_min:
                # the barrier keeps every iterate feasible, so a small
                # violation proves nothing on its own — the inner solves are
                # what decide whether this iterate is actually optimal
                success = violation <= self.tol and inner_ok
                return OptimizeResult(
                    x=x, fun=float(problem.f(x)), success=success,
                    message=(
                        f"Converged: barrier weight reached tau_min with "
                        f"violation {violation:.3g} ({inner_total} inner "
                        f"iterations)."
                        if success else
                        f"Barrier weight reached tau_min={self.tau_min:.3g} "
                        f"with violation {violation:.3g}.{inner_note}"
                    ),
                    n_iter=k + 1,
                )
            tau = max(tau * self.theta, self.tau_min)
        return OptimizeResult(
            x=x, fun=float(problem.f(x)), success=False,
            message=f"Outer iteration limit reached with violation "
                    f"{constraint_violation(problem, x):.3g}.{inner_note}",
            n_iter=self.max_iter,
        )
