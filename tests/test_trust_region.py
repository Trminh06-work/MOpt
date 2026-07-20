import numpy as np
import pytest
from scipy.optimize import minimize

from mopt.nonlinear import NLPProblem, TrustRegion, cauchy, dogleg

Q = np.array([
    [1.85, 0.03, 0.00, 0.52, 0.05],
    [0.03, 1.71, -0.22, -0.32, -0.14],
    [0.00, -0.22, 1.75, 0.14, -0.12],
    [0.52, -0.32, 0.14, 1.45, 0.18],
    [0.05, -0.14, -0.12, 0.18, 1.91],
])
M = np.array([
    [3.08, 1.28, -0.14, -0.40, 0.38],
    [1.28, 5.06, 0.91, 0.86, 0.89],
    [-0.14, 0.91, 4.77, 0.61, 0.15],
    [-0.40, 0.86, 0.61, 3.28, 1.59],
    [0.38, 0.89, 0.15, 1.59, 3.99],
])
b = np.array([1.6, 1.2, 0.9, 0.6, -0.1])
c = np.array([-0.2, 0.8, 1.4, -0.7, -0.3])


def quartic_problem():
    # f(x) = (x@Q@x + b@x)^2 + x@M@x + c@x, derivatives cross-checked
    # against finite differences in the lab notebook
    s = lambda x: x @ Q @ x + b @ x

    def f(x):
        return s(x) ** 2 + x @ M @ x + c @ x

    def grad(x):
        return 2.0 * s(x) * (2.0 * Q @ x + b) + 2.0 * M @ x + c

    def hess(x):
        g_s = 2.0 * Q @ x + b
        return 2.0 * np.outer(g_s, g_s) + 4.0 * s(x) * Q + 2.0 * M

    return NLPProblem(f=f, x0=np.zeros(5), grad=grad, hess=hess)


def rosen_problem():
    def f(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def grad(x):
        g = np.zeros_like(x)
        g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    def hess(x):
        return np.array(
            [
                [1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0, -400.0 * x[0]],
                [-400.0 * x[0], 200.0],
            ]
        )

    return NLPProblem(f=f, x0=[-1.2, 1.0], grad=grad, hess=hess)


def test_quartic_matches_scipy():
    problem = quartic_problem()
    reference = minimize(
        problem.f, problem.x0, jac=problem.grad, hess=problem.hess,
        method="trust-exact",
    )
    for method in (cauchy, dogleg):
        result = TrustRegion(method=method).solve(problem)
        assert result.success
        np.testing.assert_allclose(result.x, reference.x, atol=1e-5)
        assert result.fun == pytest.approx(reference.fun, abs=1e-8)


def test_dogleg_solves_quadratic_in_few_iterations():
    # exact model: the (clipped) Newton step finishes as soon as it fits
    problem = NLPProblem(
        f=lambda x: 0.5 * x @ M @ x + c @ x,
        x0=np.zeros(5),
        grad=lambda x: M @ x + c,
        hess=lambda x: M,
    )
    result = TrustRegion(method=dogleg).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(M, -c), atol=1e-8)
    assert result.n_iter <= 5


def test_cauchy_solves_quadratic():
    problem = NLPProblem(
        f=lambda x: 0.5 * x @ M @ x + c @ x,
        x0=np.zeros(5),
        grad=lambda x: M @ x + c,
        hess=lambda x: M,
    )
    result = TrustRegion(method=cauchy).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(M, -c), atol=1e-5)


def test_dogleg_rosenbrock():
    result = TrustRegion(method=dogleg, tol=1e-8).solve(rosen_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)


def test_requires_hessian():
    problem = NLPProblem(f=lambda x: float(x @ x), x0=[1.0, 1.0])
    with pytest.raises(ValueError, match="hess"):
        TrustRegion().solve(problem)


def test_dogleg_indefinite_hessian_fails_gracefully():
    # saddle: curvature along the gradient is zero at (1, 1)
    problem = NLPProblem(
        f=lambda x: x[0] ** 2 - x[1] ** 2,
        x0=[1.0, 1.0],
        grad=lambda x: np.array([2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: np.diag([2.0, -2.0]),
    )
    result = TrustRegion(method=dogleg).solve(problem)
    assert not result.success
    assert "Subproblem failed" in result.message


def test_iteration_limit():
    result = TrustRegion(method=cauchy, max_iter=3).solve(rosen_problem())
    assert not result.success


def test_custom_method_satisfies_protocol():
    # any callable with the (grad, hessian, radius) shape plugs in
    half_cauchy = lambda grad, hessian, radius: cauchy(grad, hessian, radius / 2)
    result = TrustRegion(method=half_cauchy).solve(
        NLPProblem(
            f=lambda x: 0.5 * x @ M @ x + c @ x,
            x0=np.zeros(5),
            grad=lambda x: M @ x + c,
            hess=lambda x: M,
        )
    )
    assert result.success
