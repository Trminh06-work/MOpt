from functools import partial

import numpy as np
import pytest

from mopt import BaseOptimizer
from mopt.nonlinear import (
    NLPProblem,
    QuasiNewton,
    armijo,
    bfgs,
    bracketing_wolfe,
    dfp,
    wolfe,
)

Q = np.array([
    [1.42, 0.06, 0.09, 0.09, 0.19],
    [0.06, 1.57, -0.24, 0.13, 0.23],
    [0.09, -0.24, 1.26, 0.11, -0.13],
    [0.09, 0.13, 0.11, 1.96, 0.02],
    [0.19, 0.23, -0.13, 0.02, 1.24],
])
c = np.array([-1.1, 0.3, -0.9, 0.4, -1.4])

UPDATES = [bfgs, dfp]


def quad_problem(x0=None):
    # f(x) = x@Q@x/2 - c@x: strictly convex, minimizer solves Q x = c
    return NLPProblem(
        f=lambda x: x @ Q @ x / 2.0 - c @ x,
        x0=np.zeros(5) if x0 is None else x0,
        grad=lambda x: Q @ x - c,
        hess=lambda x: Q,
    )


def rosen_problem():
    def f(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def grad(x):
        g = np.zeros_like(x)
        g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    return NLPProblem(f=f, x0=[-1.2, 1.0], grad=grad)


@pytest.mark.parametrize("update", UPDATES)
def test_update_satisfies_secant_equation(update):
    # the defining property: H_next @ y == s
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    H = A @ A.T + 5.0 * np.eye(5)  # symmetric positive definite
    s = rng.normal(size=5)
    y = H @ s + 0.3 * s  # keeps s @ y > 0
    H_next = update(H, s, y)
    np.testing.assert_allclose(H_next @ y, s, atol=1e-10)


@pytest.mark.parametrize("update", UPDATES)
def test_update_preserves_symmetry_and_definiteness(update):
    rng = np.random.default_rng(1)
    H = np.eye(4)
    for _ in range(20):
        s = rng.normal(size=4)
        y = rng.normal(size=4)
        if s @ y <= 0:
            y = -y  # enforce the curvature condition
        H = update(H, s, y)
        np.testing.assert_allclose(H, H.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(H) > 0)


def test_bfgs_inverse_matches_direct_formula():
    # the H update is the SMW inverse of the direct B update
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4))
    B = A @ A.T + 4.0 * np.eye(4)
    s = rng.normal(size=4)
    y = B @ s + 0.5 * s
    B_next = (
        B
        - np.outer(B @ s, B @ s) / float(s @ B @ s)
        + np.outer(y, y) / float(y @ s)
    )
    np.testing.assert_allclose(
        bfgs(np.linalg.inv(B), s, y), np.linalg.inv(B_next), atol=1e-10
    )


@pytest.mark.parametrize("update", UPDATES)
def test_quasi_newton_on_quadratic(update):
    solver = QuasiNewton(update=update)
    assert isinstance(solver, BaseOptimizer)
    result = solver.solve(quad_problem())
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-6)


def test_quasi_newton_rosenbrock_bfgs_at_defaults():
    result = QuasiNewton(update=bfgs).solve(rosen_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("update", UPDATES)
def test_quasi_newton_rosenbrock(update):
    # DFP needs a more exact line than the sigma=0.9 default; BFGS does
    # not care, which is the classic distinction between the two
    search = partial(bracketing_wolfe, sigma=0.1)
    result = QuasiNewton(update=update, line_search=search).solve(rosen_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)


def test_shrink_only_search_cannot_reach_the_window():
    # the reason bracketing_wolfe is the default: on Rosenbrock the
    # acceptable step repeatedly sits above eta=1, which wolfe can only
    # approach from above and so never reaches
    result = QuasiNewton(update=bfgs, line_search=wolfe).solve(rosen_problem())
    assert not result.success
    assert "Line search failed" in result.message


@pytest.mark.parametrize("update", UPDATES)
def test_quasi_newton_from_several_starts(update):
    rng = np.random.default_rng(3)
    x_star = np.linalg.solve(Q, c)
    for x0 in rng.uniform(-2.0, 2.0, size=(3, 5)):
        result = QuasiNewton(update=update).solve(quad_problem(x0))
        assert result.success
        np.testing.assert_allclose(result.x, x_star, atol=1e-6)


def test_quasi_newton_accepts_initial_approximation():
    # H0 = Q^-1 makes the first direction the exact Newton step
    result = QuasiNewton(H0=np.linalg.inv(Q)).solve(quad_problem())
    assert result.success
    assert result.n_iter == 1
    np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-8)


def test_quasi_newton_rejects_mismatched_h0():
    with pytest.raises(ValueError, match="H0 has shape"):
        QuasiNewton(H0=np.eye(3)).solve(quad_problem())


def test_quasi_newton_reports_iteration_limit():
    result = QuasiNewton(max_iter=2).solve(rosen_problem())
    assert not result.success
    assert "Iteration limit" in result.message
    assert result.n_iter == 2


def test_quasi_newton_reports_curvature_failure():
    # armijo alone does not imply s @ y > 0, so the guard eventually trips
    result = QuasiNewton(update=dfp, line_search=armijo).solve(rosen_problem())
    assert not result.success
    assert "Curvature condition" in result.message


def test_quasi_newton_finite_difference_fallback():
    # no analytic gradient: NLPProblem.gradient falls back to differences
    problem = NLPProblem(f=lambda x: x @ Q @ x / 2.0 - c @ x, x0=np.zeros(5))
    result = QuasiNewton(tol=1e-5).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, np.linalg.solve(Q, c), atol=1e-5)
