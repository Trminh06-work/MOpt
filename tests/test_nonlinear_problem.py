import numpy as np
import pytest

from mopt.nonlinear import NLPProblem, finite_difference_gradient


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def test_finite_diff_quadratic():
    # f(x) = x'Qx + b'x has gradient (Q + Q')x + b
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 4))
    b = rng.normal(size=4)
    x = rng.normal(size=4)
    g = finite_difference_gradient(lambda z: z @ Q @ z + b @ z, x)
    np.testing.assert_allclose(g, (Q + Q.T) @ x + b, rtol=1e-6, atol=1e-8)


def test_finite_diff_rosenbrock():
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.uniform(-2, 2, size=5)
        g = finite_difference_gradient(rosenbrock, x)
        np.testing.assert_allclose(g, rosenbrock_grad(x), rtol=1e-5, atol=1e-6)


def test_finite_diff_transcendental():
    # f(x) = sum(exp(x) + sin(x)) has gradient exp(x) + cos(x)
    x = np.array([0.0, 0.5, -1.5])
    g = finite_difference_gradient(lambda z: np.sum(np.exp(z) + np.sin(z)), x)
    np.testing.assert_allclose(g, np.exp(x) + np.cos(x), rtol=1e-6)


def test_gradient_prefers_user_grad():
    sentinel = np.array([7.0, 7.0])
    problem = NLPProblem(f=lambda x: 0.0, x0=[0, 0], grad=lambda x: sentinel)
    np.testing.assert_array_equal(problem.gradient(np.zeros(2)), sentinel)


def test_gradient_falls_back_to_finite_diff():
    problem = NLPProblem(f=rosenbrock, x0=np.zeros(4))
    x = np.array([0.5, -0.3, 1.2, 0.1])
    np.testing.assert_allclose(
        problem.gradient(x), rosenbrock_grad(x), rtol=1e-5, atol=1e-6
    )


def test_x0_normalized_to_float_array():
    problem = NLPProblem(f=lambda x: 0.0, x0=[1, 2, 3])
    assert problem.x0.dtype == np.float64
    assert problem.x0.shape == (3,)


def test_problem_validation():
    with pytest.raises(TypeError):
        NLPProblem(f=42, x0=[0.0])
    with pytest.raises(ValueError):
        NLPProblem(f=lambda x: 0.0, x0=np.zeros((2, 2)))
