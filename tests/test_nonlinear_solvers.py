from functools import partial

import numpy as np
import pytest

from mopt import BaseOptimizer
from mopt.nonlinear import GradientDescent, Newton, NLPProblem, wolfe

Q = np.array([[1.87, -0.21], [-0.21, 1.79]])


def quad_problem():
    # strictly convex, unique minimizer at the origin with f = 0
    return NLPProblem(
        f=lambda x: x @ Q @ x,
        x0=[-2.0, 3.0],
        grad=lambda x: 2.0 * (Q @ x),
        hess=lambda x: 2.0 * Q,
    )


def rosen_problem():
    # classic 2-D Rosenbrock from (-1.2, 1); minimum f = 0 at (1, 1)
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


def test_gradient_descent_quadratic():
    solver = GradientDescent()
    assert isinstance(solver, BaseOptimizer)
    result = solver.solve(quad_problem())
    assert result.success
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-5)
    assert result.fun == pytest.approx(0.0, abs=1e-10)
    assert 0 < result.n_iter < 100


def test_gradient_descent_pluggable_line_search():
    # any callable satisfying the LineSearch protocol plugs in
    weak = GradientDescent(line_search=partial(wolfe, types="weak"))
    assert weak.solve(quad_problem()).success

    fixed_step = GradientDescent(line_search=lambda f, x, d, grad_f=None: (0, 0.1))
    result = fixed_step.solve(quad_problem())
    assert result.success
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-5)


def test_gradient_descent_rosenbrock_progress_and_limit():
    # steepest descent crawls along the banana valley: huge progress,
    # no convergence to tol within the budget
    result = GradientDescent(max_iter=3000).solve(rosen_problem())
    assert not result.success
    assert "Iteration limit" in result.message
    assert result.n_iter == 3000
    assert result.fun < 1e-4  # started at 24.2


def test_newton_damped_rosenbrock():
    result = Newton(damped=True).solve(rosen_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)
    assert result.n_iter <= 30


def test_newton_undamped_rosenbrock():
    # the classic run: pure Newton from (-1.2, 1) needs only a handful
    result = Newton(damped=False).solve(rosen_problem())
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)
    assert result.n_iter <= 10


def test_newton_solves_quadratic_in_one_step():
    # H d = -g lands exactly on the minimizer of any quadratic, and the
    # unit step satisfies Armijo (gamma < 1/2), so damped matches
    for damped in (True, False):
        result = Newton(damped=damped).solve(quad_problem())
        assert result.success
        assert result.n_iter == 1
        np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-12)


def test_newton_requires_hessian():
    problem = NLPProblem(f=lambda x: float(x @ x), x0=[1.0, 1.0])
    with pytest.raises(ValueError, match="hess"):
        Newton().solve(problem)


def test_newton_singular_hessian_fails_gracefully():
    # H = diag(12 x0^2, 2) is singular at x0 = 0 while grad = (0, 2) != 0
    problem = NLPProblem(
        f=lambda x: x[0] ** 4 + x[1] ** 2,
        x0=[0.0, 1.0],
        grad=lambda x: np.array([4.0 * x[0] ** 3, 2.0 * x[1]]),
        hess=lambda x: np.diag([12.0 * x[0] ** 2, 2.0]),
    )
    result = Newton().solve(problem)
    assert not result.success
    assert "Singular Hessian" in result.message


def saddle_problem():
    # f = x0^2 - x1^2: saddle at the origin, indefinite Hessian
    return NLPProblem(
        f=lambda x: x[0] ** 2 - x[1] ** 2,
        x0=[1.0, 1.0],
        grad=lambda x: np.array([2.0 * x[0], -2.0 * x[1]]),
        hess=lambda x: np.diag([2.0, -2.0]),
    )


def test_newton_damped_rejects_non_descent_direction():
    result = Newton(damped=True).solve(saddle_problem())
    assert not result.success
    assert "not a descent direction" in result.message


def test_newton_undamped_saddle_hazard():
    # documented hazard: with no safeguards, pure Newton happily jumps
    # straight to the saddle point and reports a zero gradient
    result = Newton(damped=False).solve(saddle_problem())
    assert result.success
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-12)
