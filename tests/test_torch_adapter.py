import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mopt.autodiff import from_torch, torch_function, torch_gradient
from mopt.nonlinear import NLPProblem, finite_difference_gradient


def rosenbrock_t(x):
    return torch.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def test_torch_gradient_matches_analytic():
    grad = torch_gradient(rosenbrock_t)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(-2, 2, size=6)
        # autodiff is exact, so only rounding error is allowed
        np.testing.assert_allclose(grad(x), rosenbrock_grad(x), rtol=1e-12)


def test_torch_gradient_is_float64():
    g = torch_gradient(rosenbrock_t)(np.zeros(3))
    assert g.dtype == np.float64


def test_finite_diff_cross_checked_against_torch():
    def f_np(x):
        return float(np.sum(np.exp(-(x**2)) + np.log1p(x**2) * np.cos(x)))

    def f_t(x):
        return torch.sum(torch.exp(-(x**2)) + torch.log1p(x**2) * torch.cos(x))

    rng = np.random.default_rng(1)
    exact = torch_gradient(f_t)
    for _ in range(5):
        x = rng.uniform(-3, 3, size=4)
        approx = finite_difference_gradient(f_np, x)
        np.testing.assert_allclose(approx, exact(x), rtol=1e-5, atol=1e-7)


def test_torch_function_returns_python_float():
    f = torch_function(rosenbrock_t)
    value = f(np.array([1.0, 1.0, 1.0]))
    assert isinstance(value, float)
    assert value == pytest.approx(0.0)  # Rosenbrock minimum at all-ones


def test_from_torch_builds_ready_problem():
    problem = from_torch(rosenbrock_t, x0=np.zeros(4))
    assert isinstance(problem, NLPProblem)
    assert problem.grad is not None  # exact gradient wired in, no FD fallback
    x = np.array([0.5, -0.3, 1.2, 0.1])
    assert problem.f(x) == pytest.approx(rosenbrock_t(torch.tensor(x)).item())
    np.testing.assert_allclose(problem.gradient(x), rosenbrock_grad(x), rtol=1e-12)
