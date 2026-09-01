import numpy as np
import pytest
from scipy.optimize import minimize

from mopt.nonlinear import SQP, ConstrainedNLPProblem, damped_bfgs, kkt_residual

# Lab test problem: min (x'Qx + b'x)^2  s.t.  x'Mx <= 26.
Q = np.array([
    [1.7, -0.8, -0.3, 0.7, -0.3],
    [-0.8, 5.0, 1.7, -0.5, -1.5],
    [-0.3, 1.7, 2.2, 0.1, -1.1],
    [0.7, -0.5, 0.1, 3.1, -1.2],
    [-0.3, -1.5, -1.1, -1.2, 2.8],
])
b = np.array([3.0, -5.0, -5.0, -1.0, 1.0])
M = np.array([
    [5.5, 1.3, -0.7, 0.0, 1.0],
    [1.3, 3.1, -1.3, 1.4, -0.2],
    [-0.7, -1.3, 5.1, 1.4, -1.5],
    [0.0, 1.4, 1.4, 3.7, -1.9],
    [1.0, -0.2, -1.5, -1.9, 4.8],
])


def lab_problem(x0):
    return ConstrainedNLPProblem(
        f=lambda x: float((x @ Q @ x + b @ x) ** 2),
        x0=np.asarray(x0, dtype=float),
        grad=lambda x: 2.0 * (2.0 * (Q @ x) + b) * (x @ Q @ x + b @ x),
        ineq=lambda x: np.array([x @ M @ x - 26.0]),
        ineq_jac=lambda x: (2.0 * (M @ x))[None, :],
    )


def equality_problem():
    # min x'x s.t. x0 + x1 = 2; minimizer (1, 1)
    return ConstrainedNLPProblem(
        f=lambda x: float(x @ x),
        x0=np.array([3.0, -1.0]),
        grad=lambda x: 2.0 * x,
        eq=lambda x: np.array([x[0] + x[1] - 2.0]),
        eq_jac=lambda x: np.array([[1.0, 1.0]]),
    )


def active_inequality_problem():
    # min x'x s.t. x0 >= 1; minimizer (1, 0) with the constraint active
    return ConstrainedNLPProblem(
        f=lambda x: float(x @ x),
        x0=np.array([3.0, 2.0]),
        grad=lambda x: 2.0 * x,
        ineq=lambda x: np.array([1.0 - x[0]]),
        ineq_jac=lambda x: np.array([[-1.0, 0.0]]),
    )


def mixed_problem():
    # min |x - (2,2)|^2 s.t. |x| <= 1 and x0 = x1; minimizer (1,1)/sqrt(2)
    return ConstrainedNLPProblem(
        f=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2),
        x0=np.zeros(2),
        grad=lambda x: 2.0 * (x - np.array([2.0, 2.0])),
        ineq=lambda x: np.array([x @ x - 1.0]),
        ineq_jac=lambda x: 2.0 * x[None, :],
        eq=lambda x: np.array([x[0] - x[1]]),
        eq_jac=lambda x: np.array([[1.0, -1.0]]),
    )


def test_damped_bfgs_reduces_to_bfgs_when_curvature_is_good():
    B = np.eye(3)
    s = np.array([1.0, 0.0, 0.0])
    y = np.array([2.0, 0.0, 0.0])  # s @ y = 2 >= 0.2 * s @ B @ s
    updated = damped_bfgs(B, s, y)
    # undamped, so the secant equation holds exactly
    np.testing.assert_allclose(updated @ s, y, atol=1e-12)


def test_damped_bfgs_stays_positive_definite_on_negative_curvature():
    B = np.eye(3)
    s = np.array([1.0, 0.0, 0.0])
    y = np.array([-5.0, 0.0, 0.0])  # s @ y < 0: plain BFGS would break
    updated = damped_bfgs(B, s, y, damping=0.2)
    assert np.linalg.eigvalsh(updated).min() > 0
    # damping targets s @ y_hat = damping * s @ B @ s exactly
    assert float(s @ (updated @ s)) == pytest.approx(0.2, rel=1e-12)


def test_damped_bfgs_validation():
    B, s, y = np.eye(2), np.array([1.0, 0.0]), np.array([1.0, 0.0])
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="damping"):
            damped_bfgs(B, s, y, damping=bad)
    with pytest.raises(ValueError, match="positive definite"):
        damped_bfgs(B, np.zeros(2), y)


def test_matches_scipy_from_every_lab_start():
    rng = np.random.default_rng(42)
    starts = np.vstack([np.ones(5), rng.uniform(-2.0, 2.0, size=(10, 5))])
    for x0 in starts:
        problem = lab_problem(x0)
        reference = minimize(
            problem.f, x0, jac=problem.grad, method="SLSQP", tol=1e-12,
            constraints=[{"type": "ineq", "fun": lambda z: -problem.ineq(z)}],
        )
        result = SQP().solve(problem)
        assert result.success, result.message
        assert result.fun == pytest.approx(reference.fun, abs=1e-6)
        assert problem.ineq(result.x)[0] <= 1e-8  # feasible
        assert result.n_iter < 50


def test_certifies_kkt_at_the_solution():
    problem = lab_problem(np.ones(5))
    result = SQP().solve(problem)
    assert result.success
    assert kkt_residual(problem, result.x).satisfied(1e-5)


def test_equality_constrained_problem():
    problem = equality_problem()
    result = SQP().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-7)
    assert result.fun == pytest.approx(2.0, abs=1e-7)


def test_active_inequality_recovers_multiplier():
    problem = active_inequality_problem()
    result = SQP().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-7)
    residual = kkt_residual(problem, result.x)
    assert residual.active[0]  # the constraint is binding
    assert residual.lam[0] == pytest.approx(2.0, abs=1e-5)  # grad f = -lam * grad g


def test_mixed_equality_and_inequality():
    problem = mixed_problem()
    result = SQP().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, np.ones(2) / np.sqrt(2.0), atol=1e-6)


def test_random_battery_against_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 6))
        A = rng.normal(size=(n, n))
        Qr = A.T @ A + 0.5 * np.eye(n)
        br = rng.normal(size=n)
        C = rng.normal(size=(n, n))
        Mr = C.T @ C + 0.5 * np.eye(n)
        a = rng.normal(size=n)
        rhs = float(rng.normal())
        problem = ConstrainedNLPProblem(
            f=lambda x, Qr=Qr, br=br: float(x @ Qr @ x + br @ x),
            x0=rng.uniform(-1.0, 1.0, size=n),
            grad=lambda x, Qr=Qr, br=br: 2.0 * (Qr @ x) + br,
            ineq=lambda x, Mr=Mr: np.array([x @ Mr @ x - 5.0]),
            ineq_jac=lambda x, Mr=Mr: (2.0 * (Mr @ x))[None, :],
            eq=lambda x, a=a, rhs=rhs: np.array([a @ x - rhs]),
            eq_jac=lambda x, a=a: a[None, :],
        )
        reference = minimize(
            problem.f, problem.x0, jac=problem.grad, method="SLSQP", tol=1e-12,
            constraints=[
                {"type": "eq", "fun": problem.eq},
                {"type": "ineq", "fun": lambda z: -problem.ineq(z)},
            ],
        )
        if not reference.success:
            continue
        result = SQP().solve(problem)
        assert result.success, result.message
        assert result.fun == pytest.approx(reference.fun, rel=1e-6, abs=1e-8)


def test_starts_from_an_infeasible_point():
    # unlike the barrier method, SQP does not need a feasible x0
    problem = lab_problem(np.full(5, 4.0))
    assert problem.ineq(problem.x0)[0] > 0
    result = SQP().solve(problem)
    assert result.success
    assert problem.ineq(result.x)[0] <= 1e-8


def test_requires_algebraic_constraints():
    problem = ConstrainedNLPProblem(
        f=lambda x: float(x @ x), x0=np.ones(2), grad=lambda x: 2.0 * x
    )
    with pytest.raises(ValueError, match="ineq"):
        SQP().solve(problem)


def test_parameter_validation():
    with pytest.raises(ValueError, match="gamma"):
        SQP(gamma=0.5)
    with pytest.raises(ValueError, match="delta"):
        SQP(delta=1.0)


def test_ignores_the_hessian_it_is_given():
    # the damped BFGS model replaces it, so a wrong hess must not matter
    problem = equality_problem()
    problem.hess = lambda x: np.full((2, 2), np.nan)
    result = SQP().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-7)
