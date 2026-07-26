import numpy as np
import pytest

from mopt import BaseOptimizer
from mopt.nonlinear import (
    ArmijoModifiedCG,
    ConjugateGradient,
    NLPProblem,
    QuadraticCG,
    fletcher_reeves,
    polak_ribiere,
)

Q = np.array([
    [1.42, 0.06, 0.09, 0.09, 0.19],
    [0.06, 1.57, -0.24, 0.13, 0.23],
    [0.09, -0.24, 1.26, 0.11, -0.13],
    [0.09, 0.13, 0.11, 1.96, 0.02],
    [0.19, 0.23, -0.13, 0.02, 1.24],
])
M = np.array([
    [7.16, 0.26, 0.30, 1.01, 0.06],
    [0.26, 6.54, -0.44, -0.71, -0.04],
    [0.30, -0.44, 5.87, -0.44, -0.24],
    [1.01, -0.71, -0.44, 6.06, 2.48],
    [0.06, -0.04, -0.24, 2.48, 4.04],
])
b = np.array([1.7, -0.7, -1.9, 0.8, -2.0])
c = np.array([-1.1, 0.3, -0.9, 0.4, -1.4])

NONLINEAR_SOLVERS = [ConjugateGradient, ArmijoModifiedCG]


def quad_problem(x0=None):
    # f(x) = x@Q@x/2 - c@x: strictly convex, minimizer solves Q x = c
    return NLPProblem(
        f=lambda x: x @ Q @ x / 2.0 - c @ x,
        x0=np.zeros(5) if x0 is None else x0,
        grad=lambda x: Q @ x - c,
        hess=lambda x: Q,
    )


def quartic_problem(x0=None):
    # same quartic as the lab notebook; derivatives cross-checked there
    # against finite differences
    s = lambda x: x @ Q @ x + b @ x

    def f(x):
        return s(x) ** 2 + x @ M @ x + c @ x

    def grad(x):
        return (4.0 * Q @ x + 2.0 * b) * s(x) + 2.0 * M @ x + c

    return NLPProblem(f=f, x0=np.zeros(5) if x0 is None else x0, grad=grad)


def rosen_problem():
    def f(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def grad(x):
        g = np.zeros_like(x)
        g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    return NLPProblem(f=f, x0=[-1.2, 1.0], grad=grad)


def test_quadratic_cg_terminates_within_n_iterations():
    # the defining property of linear CG: exact minimizer in at most n steps
    solver = QuadraticCG()
    assert isinstance(solver, BaseOptimizer)
    result = solver.solve(quad_problem())
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-10)
    assert result.n_iter <= 5


def test_quadratic_cg_requires_hessian():
    problem = NLPProblem(f=lambda x: float(x @ x), x0=[1.0, 1.0])
    with pytest.raises(ValueError, match="hess"):
        QuadraticCG().solve(problem)


def test_quadratic_cg_rejects_indefinite_q():
    # f = x0^2 - x1^2 has an indefinite (constant) Hessian
    problem = NLPProblem(
        f=lambda x: x[0] ** 2 - x[1] ** 2,
        x0=[1.0, 1.0],
        grad=lambda x: np.array([2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: np.diag([2.0, -2.0]),
    )
    result = QuadraticCG().solve(problem)
    assert not result.success
    assert "Nonpositive curvature" in result.message


@pytest.mark.parametrize("solver_cls", NONLINEAR_SOLVERS)
def test_nonlinear_cg_on_quadratic(solver_cls):
    solver = solver_cls()
    assert isinstance(solver, BaseOptimizer)
    result = solver.solve(quad_problem())
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-6)


@pytest.mark.parametrize("solver_cls", NONLINEAR_SOLVERS)
def test_nonlinear_cg_rosenbrock(solver_cls):
    result = solver_cls().solve(rosen_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("solver_cls", NONLINEAR_SOLVERS)
def test_nonlinear_cg_quartic_from_several_starts(solver_cls):
    # every start reaches the same minimum; f value from the lab notebook
    rng = np.random.default_rng(1)
    for x0 in np.vstack([np.zeros(5), rng.uniform(-2.0, 2.0, size=(3, 5))]):
        result = solver_cls().solve(quartic_problem(x0))
        assert result.success
        assert result.fun == pytest.approx(-0.24514746, abs=1e-6)


@pytest.mark.parametrize("solver_cls", NONLINEAR_SOLVERS)
def test_nonlinear_cg_reports_iteration_limit(solver_cls):
    result = solver_cls(max_iter=2).solve(rosen_problem())
    assert not result.success
    assert "Iteration limit" in result.message
    assert result.n_iter == 2


@pytest.mark.parametrize("solver_cls", NONLINEAR_SOLVERS)
def test_nonlinear_cg_accepts_pluggable_beta(solver_cls):
    for beta in (fletcher_reeves, polak_ribiere):
        result = solver_cls(beta=beta).solve(quad_problem())
        assert result.success
        np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-6)


@pytest.mark.parametrize("solver_cls", NONLINEAR_SOLVERS)
def test_nonlinear_cg_finite_difference_fallback(solver_cls):
    # no analytic gradient: NLPProblem.gradient falls back to differences
    problem = NLPProblem(f=lambda x: x @ Q @ x / 2.0 - c @ x, x0=np.zeros(5))
    result = solver_cls(tol=1e-5).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-5)


def test_armijo_modified_validates_tunables():
    for kwargs in (
        {"rho_1": 2.0, "rho_2": 1.0},
        {"rho_1": -1.0},
        {"gamma": 1.0},
        {"delta": 0.0},
        {"theta": 1.0},
    ):
        with pytest.raises(ValueError):
            ArmijoModifiedCG(**kwargs)


def test_armijo_modified_condition_ii_binds():
    # (ii) must reject steps that (i) alone accepts, otherwise the extra
    # test is a no-op; count them directly on the Rosenbrock run
    problem = rosen_problem()
    solver = ArmijoModifiedCG()
    x = problem.x0.copy()
    g = problem.gradient(x)
    d = -g
    rejected_by_ii = 0
    for _ in range(50):
        fx, slope = float(problem.f(x)), float(g @ d)
        eta = (solver.rho_1 + solver.rho_2) / 2 * abs(slope) / float(d @ d)
        while True:
            x_next = x + eta * d
            g_next = problem.gradient(x_next)
            gg_next = float(g_next @ g_next)
            d_next = -g_next + solver.beta(g_next, g, d) * d
            if problem.f(x_next) <= fx + solver.gamma * eta * slope:
                if float(g_next @ d_next) < -solver.delta * gg_next:
                    break
                rejected_by_ii += 1  # (i) passed, (ii) did not
            eta *= solver.theta
        x, g, d = x_next, g_next, d_next
    assert rejected_by_ii > 0
