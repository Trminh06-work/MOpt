import numpy as np
import pytest
from scipy.optimize import minimize

from mopt.nonlinear import (
    ConstrainedNLPProblem,
    GradientDescent,
    PenaltyBarrierMethod,
    PenaltyMethod,
    TrustRegion,
    barrier_merit,
    cauchy,
    constraint_violation,
    finite_difference_gradient,
    penalty_merit,
)

# Lab test problem: min x'Qx + b'x  s.t.  c'x = -10,  x'Mx <= 82,
# with Q and M symmetric positive definite.
Q = np.array([
    [5.9, -0.1, 0.1, 0.3, -1.1],
    [-0.1, 7.9, -0.2, -0.3, -0.1],
    [0.1, -0.2, 6.9, 0.1, -0.3],
    [0.3, -0.3, 0.1, 8.4, 0.1],
    [-1.1, -0.1, -0.3, 0.1, 6.8],
])
b = np.array([3.0, -3.0, -2.0, -2.0, 1.0])
c = np.array([-4.0, -1.0, 5.0, 4.0, 1.0])
M = np.array([
    [5.8, -1.3, -1.2, 0.2, 1.2],
    [-1.3, 3.5, -1.9, -1.0, -1.8],
    [-1.2, -1.9, 4.4, 1.3, 1.3],
    [0.2, -1.0, 1.3, 4.1, -2.0],
    [1.2, -1.8, 1.3, -2.0, 6.1],
])


def lab_problem(x0=None):
    return ConstrainedNLPProblem(
        f=lambda x: float(x @ Q @ x + b @ x),
        x0=np.zeros(5) if x0 is None else x0,
        grad=lambda x: 2.0 * (Q @ x) + b,
        hess=lambda x: 2.0 * Q,
        ineq=lambda x: np.array([x @ M @ x - 82.0]),
        ineq_jac=lambda x: (2.0 * (M @ x))[None, :],
        eq=lambda x: np.array([c @ x + 10.0]),
        eq_jac=lambda x: c[None, :],
    )


def lab_reference():
    problem = lab_problem()
    return minimize(
        problem.f, problem.x0, jac=problem.grad, method="SLSQP", tol=1e-12,
        constraints=[
            {"type": "eq", "fun": problem.eq},
            {"type": "ineq", "fun": lambda z: -problem.ineq(z)},
        ],
    )


def equality_problem(x0=(2.0, 2.0)):
    # min x'x s.t. x0 + x1 = 2; minimizer (1, 1), f = 2
    return ConstrainedNLPProblem(
        f=lambda x: float(x @ x),
        x0=np.asarray(x0, dtype=float),
        grad=lambda x: 2.0 * x,
        hess=lambda x: 2.0 * np.eye(2),
        eq=lambda x: np.array([x[0] + x[1] - 2.0]),
        eq_jac=lambda x: np.array([[1.0, 1.0]]),
    )


def test_constraint_violation_measures_both_kinds():
    problem = lab_problem()
    feasible = np.array([0.0, 0.0, -2.0, 0.0, 0.0])  # c'x + 10 = 0, x'Mx << 82
    assert constraint_violation(problem, feasible) == pytest.approx(0.0, abs=1e-12)
    # equality off by 3 dominates the satisfied inequality
    assert constraint_violation(problem, np.zeros(5)) == pytest.approx(10.0)


def test_penalty_merit_gradient_matches_finite_differences():
    problem = lab_problem()
    x = np.array([0.3, -0.2, 0.1, 0.4, -0.1])
    for tau in (0.5, 3.0, 50.0):
        merit = penalty_merit(problem, tau)
        analytic = merit.grad(x)
        approx = finite_difference_gradient(merit.f, x)
        np.testing.assert_allclose(approx, analytic, rtol=1e-6)


def test_barrier_merit_gradient_matches_finite_differences():
    problem = lab_problem()
    x = np.array([0.3, -0.2, 0.1, 0.4, -0.1])
    for tau in (0.5, 3.0):
        merit = barrier_merit(problem, tau)
        analytic = merit.grad(x)
        approx = finite_difference_gradient(merit.f, x)
        np.testing.assert_allclose(approx, analytic, rtol=1e-6)


def test_penalty_merit_hessian_exact_for_affine_constraints():
    # the Gauss-Newton Hessian drops constraint curvature, so it is the
    # true Hessian exactly when the constraints are affine
    problem = equality_problem()
    merit = penalty_merit(problem, 3.0)
    x = np.array([0.3, -0.4])
    h = 1e-6
    approx = np.column_stack([
        (merit.grad(x + h * e) - merit.grad(x - h * e)) / (2.0 * h)
        for e in np.eye(2)
    ])
    np.testing.assert_allclose(approx, merit.hess(x), atol=1e-6)


def test_barrier_merit_is_infinite_outside_feasible_set():
    problem = lab_problem()
    merit = barrier_merit(problem, 1.0)
    outside = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # x'Mx far above 82
    assert problem.ineq(outside)[0] > 0
    assert np.isinf(merit.f(outside))


def test_penalty_method_matches_scipy_on_lab_problem():
    problem = lab_problem()
    reference = lab_reference()
    result = PenaltyMethod().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, reference.x, atol=1e-6)
    assert result.fun == pytest.approx(reference.fun, abs=1e-6)
    assert constraint_violation(problem, result.x) <= 1e-6


def test_penalty_barrier_method_matches_scipy_on_lab_problem():
    problem = lab_problem()
    reference = lab_reference()
    result = PenaltyBarrierMethod().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, reference.x, atol=1e-6)
    assert result.fun == pytest.approx(reference.fun, abs=1e-6)


def test_barrier_iterates_stay_strictly_feasible():
    problem = lab_problem()
    result = PenaltyBarrierMethod().solve(problem)
    # the barrier is +inf outside, so the answer must be strictly inside
    assert problem.ineq(result.x)[0] < 0


def test_penalty_method_solves_equality_only_problem():
    result = PenaltyMethod().solve(equality_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)


def test_inner_solver_is_pluggable():
    # any BaseOptimizer works; a weaker one still lands on the right answer
    result = PenaltyMethod(
        inner_solver=GradientDescent(max_iter=2000)
    ).solve(equality_problem())
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)


def test_falls_back_to_quasi_newton_without_a_hessian():
    problem = ConstrainedNLPProblem(
        f=lambda x: float(x @ x),
        x0=np.array([2.0, 2.0]),
        grad=lambda x: 2.0 * x,
        eq=lambda x: np.array([x[0] + x[1] - 2.0]),
        eq_jac=lambda x: np.array([[1.0, 1.0]]),
    )
    assert problem.hess is None
    result = PenaltyMethod().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)


def test_finite_difference_jacobian_fallback():
    # no ineq_jac / eq_jac supplied: the problem differences them itself
    problem = ConstrainedNLPProblem(
        f=lambda x: float(x @ x),
        x0=np.array([2.0, 2.0]),
        grad=lambda x: 2.0 * x,
        hess=lambda x: 2.0 * np.eye(2),
        eq=lambda x: np.array([x[0] + x[1] - 2.0]),
    )
    result = PenaltyMethod().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)


def test_barrier_does_not_claim_success_when_inner_solves_fail():
    # linear objective, and grad(ineq) vanishes at the origin, so the
    # merit Hessian is singular there and dogleg cannot start. Every
    # iterate stays feasible, so feasibility alone must not read as
    # convergence.
    problem = ConstrainedNLPProblem(
        f=lambda x: float(x[0]),
        x0=np.zeros(2),
        grad=lambda x: np.array([1.0, 0.0]),
        hess=lambda x: np.zeros((2, 2)),
        ineq=lambda x: np.array([x @ x - 1.0]),
        ineq_jac=lambda x: 2.0 * x[None, :],
    )
    result = PenaltyBarrierMethod().solve(problem)
    assert not result.success
    assert "did not converge" in result.message
    assert constraint_violation(problem, result.x) == pytest.approx(0.0)

    # the documented remedy: Cauchy points tolerate a singular model
    recovered = PenaltyBarrierMethod(
        inner_solver=TrustRegion(method=cauchy, max_iter=500)
    ).solve(problem)
    np.testing.assert_allclose(recovered.x, [-1.0, 0.0], atol=1e-3)


def test_penalty_requires_algebraic_constraints():
    problem = ConstrainedNLPProblem(
        f=lambda x: float(x @ x), x0=np.ones(2), grad=lambda x: 2.0 * x
    )
    with pytest.raises(ValueError, match="ineq"):
        PenaltyMethod().solve(problem)


def test_barrier_requires_inequality_constraints():
    with pytest.raises(ValueError, match="ineq"):
        PenaltyBarrierMethod().solve(equality_problem())


def test_barrier_requires_strictly_feasible_start():
    problem = lab_problem(x0=np.array([4.0, 0.0, 0.0, 0.0, 0.0]))
    assert problem.ineq(problem.x0)[0] > 0  # outside the ellipsoid
    with pytest.raises(ValueError, match="strictly feasible"):
        PenaltyBarrierMethod().solve(problem)


def test_parameter_validation():
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError, match="theta"):
            PenaltyMethod(theta=bad)
        with pytest.raises(ValueError, match="theta"):
            PenaltyBarrierMethod(theta=bad)
    with pytest.raises(ValueError, match="tau0"):
        PenaltyMethod(tau0=0.0)
    with pytest.raises(ValueError, match="tau"):
        penalty_merit(lab_problem(), 0.0)
    with pytest.raises(ValueError, match="tau"):
        barrier_merit(lab_problem(), -1.0)
