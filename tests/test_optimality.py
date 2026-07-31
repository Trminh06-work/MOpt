import numpy as np
import pytest

from mopt.nonlinear import (
    ConstrainedNLPProblem,
    ProjectedGradient,
    affine_projection,
    ellipsoid_oracle,
    ellipsoid_projection,
    frank_wolfe_gap,
    intersection_projection,
    kkt_residual,
    lagrangian,
    projected_gradient_residual,
)

# the lab problem: min x'Qx + b'x  s.t.  c'x = 23,  x'Mx <= 167
Q = np.array([
    [7.0, -0.2, 0.1, -0.3, 0.0],
    [-0.2, 6.9, 0.4, 0.0, 0.1],
    [0.1, 0.4, 6.6, 0.3, 0.0],
    [-0.3, 0.0, 0.3, 6.7, 0.1],
    [0.0, 0.1, 0.0, 0.1, 6.9],
])
M = np.array([
    [5.0, 0.4, -0.1, 0.1, 0.5],
    [0.4, 6.0, -0.5, -2.5, 1.8],
    [-0.1, -0.5, 4.7, 0.1, -1.5],
    [0.1, -2.5, 0.1, 4.5, -0.3],
    [0.5, 1.8, -1.5, -0.3, 3.7],
])
b = np.array([4.0, -3.0, -3.0, 2.0, 2.0])
c = np.array([-4.0, 1.0, 2.0, 4.0, -5.0])
S_ELL, S_EQ = 167.0, 23.0

# closed-form primal-dual solution of the equality-constrained problem, from
# the KKT block system [[2Q, c], [c', 0]] [x; nu] = [-b; 23]
_SOL = np.linalg.solve(
    np.block([[2 * Q, c[:, None]], [c[None, :], np.zeros((1, 1))]]),
    np.concatenate([-b, [S_EQ]]),
)
X_STAR, NU_STAR = _SOL[:5], float(_SOL[5])


def lab_problem(x0=None, algebraic=True):
    kwargs = {}
    if algebraic:
        kwargs = dict(
            ineq=lambda x: np.array([x @ M @ x - S_ELL]),
            eq=lambda x: np.array([c @ x - S_EQ]),
            ineq_jac=lambda x: (2.0 * M @ x)[None, :],
            eq_jac=lambda x: c[None, :],
        )
    return ConstrainedNLPProblem(
        f=lambda x: x @ Q @ x + b @ x,
        x0=np.zeros(5) if x0 is None else x0,
        grad=lambda x: 2.0 * Q @ x + b,
        projection=intersection_projection([
            ellipsoid_projection(M, S_ELL),
            affine_projection(c[None, :], [S_EQ]),
        ]),
        **kwargs,
    )


# --- oracle-form conditions --------------------------------------------

def test_projected_gradient_residual_vanishes_at_the_solution():
    problem = lab_problem()
    result = ProjectedGradient(alpha=0.05).solve(problem)
    assert result.success
    # measured with the step the solver used, this is what it drove below tol
    assert projected_gradient_residual(problem, result.x, alpha=0.05) < 1e-6
    # and is clearly nonzero at a feasible but suboptimal point
    elsewhere = problem.project(np.array([3.0, 1.0, -2.0, 0.5, 1.0]))
    assert projected_gradient_residual(problem, elsewhere, alpha=0.05) > 1e-3


@pytest.mark.parametrize("alpha", [0.05, 1.0, 5.0])
def test_projected_gradient_residual_zero_set_is_alpha_independent(alpha):
    # the magnitude scales with alpha, so compare against a suboptimal point
    # at the same alpha rather than against a fixed threshold
    problem = lab_problem()
    x_opt = ProjectedGradient(alpha=0.05).solve(problem).x
    elsewhere = problem.project(np.array([3.0, 1.0, -2.0, 0.5, 1.0]))
    at_opt = projected_gradient_residual(problem, x_opt, alpha=alpha)
    at_other = projected_gradient_residual(problem, elsewhere, alpha=alpha)
    assert at_opt < at_other / 100.0


def test_projected_gradient_residual_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        projected_gradient_residual(lab_problem(), np.zeros(5), alpha=0.0)


def test_frank_wolfe_gap_vanishes_at_the_solution():
    # ellipsoid only, tight enough that the constraint binds
    s = 1.0
    problem = ConstrainedNLPProblem(
        f=lambda x: x @ Q @ x + b @ x, x0=np.zeros(5),
        grad=lambda x: 2.0 * Q @ x + b,
        projection=ellipsoid_projection(M, s), lmo=ellipsoid_oracle(M, s),
    )
    result = ProjectedGradient(alpha=0.05, max_iter=5000).solve(problem)
    assert result.success
    gap = frank_wolfe_gap(problem, result.x)
    assert 0 <= gap < 1e-4
    assert frank_wolfe_gap(problem, problem.project(np.ones(5))) > 1e-3


def test_frank_wolfe_gap_requires_lmo():
    with pytest.raises(ValueError, match="lmo"):
        frank_wolfe_gap(lab_problem(), np.zeros(5))


# --- Lagrangian ---------------------------------------------------------

def test_lagrangian_reduces_to_f_with_zero_multipliers():
    problem = lab_problem()
    x = np.array([1.0, 0.0, -1.0, 2.0, 0.5])
    assert lagrangian(problem, x) == pytest.approx(problem.f(x))


def test_lagrangian_adds_the_weighted_constraints():
    problem = lab_problem()
    x = np.array([1.0, 0.0, -1.0, 2.0, 0.5])
    lam, nu = np.array([2.0]), np.array([-3.0])
    expected = (problem.f(x)
                + 2.0 * (x @ M @ x - S_ELL)
                + (-3.0) * (c @ x - S_EQ))
    assert lagrangian(problem, x, lam, nu) == pytest.approx(expected)


def test_lagrangian_is_stationary_in_x_at_the_solution():
    # the defining property: with the right multipliers, x* is a critical
    # point of L, so nudging x in any direction cannot decrease it much
    problem = lab_problem()
    base = lagrangian(problem, X_STAR, np.zeros(1), np.array([NU_STAR]))
    rng = np.random.default_rng(0)
    for _ in range(5):
        step = 1e-5 * rng.normal(size=5)
        moved = lagrangian(problem, X_STAR + step, np.zeros(1),
                           np.array([NU_STAR]))
        assert abs(moved - base) < 1e-8  # first-order term vanishes


# --- KKT ----------------------------------------------------------------

def test_kkt_holds_at_the_solution_and_recovers_the_multiplier():
    problem = lab_problem()
    residual = kkt_residual(problem, X_STAR)
    assert residual.satisfied(1e-6)
    assert residual.max_violation < 1e-6
    # the ellipsoid is slack here (x*'Mx* = 40.1 << 167), so it must be
    # inactive and carry a zero multiplier
    assert not residual.active[0]
    assert residual.lam[0] == pytest.approx(0.0)
    assert residual.nu[0] == pytest.approx(NU_STAR, rel=1e-8)


def test_kkt_fails_away_from_the_solution():
    problem = lab_problem()
    off = problem.project(np.array([3.0, 1.0, -2.0, 0.5, 1.0]))
    assert not kkt_residual(problem, off).satisfied(1e-6)


def test_kkt_agrees_with_the_solver():
    problem = lab_problem()
    result = ProjectedGradient(alpha=0.05).solve(problem)
    assert result.success
    assert kkt_residual(problem, result.x).satisfied(1e-4)


def test_kkt_flags_primal_infeasibility():
    problem = lab_problem()
    residual = kkt_residual(problem, np.zeros(5))  # c @ 0 = 0, not 23
    assert residual.primal_feasibility == pytest.approx(S_EQ)


def test_kkt_flags_negative_multiplier_as_dual_infeasible():
    # min (x - 0.5)^2 s.t. x <= 1: the optimum is interior at x = 0.5, so
    # the boundary point x = 1 is stationary only with lambda = -1
    problem = ConstrainedNLPProblem(
        f=lambda x: float((x[0] - 0.5) ** 2), x0=np.array([0.0]),
        grad=lambda x: np.array([2.0 * (x[0] - 0.5)]),
        ineq=lambda x: np.array([x[0] - 1.0]),
        ineq_jac=lambda x: np.array([[1.0]]),
    )
    residual = kkt_residual(problem, np.array([1.0]))
    assert residual.lam[0] == pytest.approx(-1.0)
    assert residual.dual_feasibility == pytest.approx(1.0)
    assert not residual.satisfied()
    # while the true minimizer passes
    assert kkt_residual(problem, np.array([0.5])).satisfied()


def test_kkt_flags_complementarity_violation():
    # a multiplier on a strictly inactive constraint
    problem = lab_problem()
    residual = kkt_residual(problem, X_STAR, lam=np.array([1.0]),
                            nu=np.array([NU_STAR]))
    assert residual.complementarity > 100.0  # |1.0 * (40.1 - 167)|
    assert not residual.satisfied()


def test_kkt_estimates_multipliers_only_over_active_constraints():
    # an inactive constraint must not be allowed to absorb a stationarity
    # violation, so its multiplier is pinned to zero
    problem = lab_problem()
    off = problem.project(np.array([3.0, 1.0, -2.0, 0.5, 1.0]))
    residual = kkt_residual(problem, off)
    assert not residual.active[0]
    assert residual.lam[0] == 0.0
    assert residual.stationarity > 1e-3


def test_kkt_uses_finite_difference_jacobians():
    # same problem without analytic constraint Jacobians
    problem = ConstrainedNLPProblem(
        f=lambda x: x @ Q @ x + b @ x, x0=np.zeros(5),
        grad=lambda x: 2.0 * Q @ x + b,
        ineq=lambda x: np.array([x @ M @ x - S_ELL]),
        eq=lambda x: np.array([c @ x - S_EQ]),
    )
    residual = kkt_residual(problem, X_STAR)
    assert residual.satisfied(1e-5)
    assert residual.nu[0] == pytest.approx(NU_STAR, rel=1e-5)


def test_kkt_requires_algebraic_constraints():
    # an oracle identifies the set but not the individual constraints
    with pytest.raises(ValueError, match="algebraic constraints"):
        kkt_residual(lab_problem(algebraic=False), X_STAR)


def test_kkt_residual_reports_max_violation():
    residual = kkt_residual(lab_problem(), np.zeros(5))
    assert residual.max_violation == max(
        residual.stationarity, residual.primal_feasibility,
        residual.dual_feasibility, residual.complementarity,
    )
