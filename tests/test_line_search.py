import numpy as np
import pytest

from mopt.nonlinear import armijo, wolfe

# Coursework quadratic with hand-verified ground truth: from x0 along d0,
# Armijo (gamma=0.14) accepts eta <= 0.4057, the strong-Wolfe window is
# [0.191, 0.281], so backtracking 0.87^k accepts k=7 (Armijo/weak) and
# k=10 (strong).
Q = np.array([[1.87, -0.21], [-0.21, 1.79]])
x0 = np.array([-2.0, 3.0])
d0 = np.array([7.95, -13.09])


def f_q(x):
    return x @ Q @ x


def grad_q(x):
    return 2.0 * (Q @ x)


def test_armijo_accepts_known_step():
    num_iter, eta = armijo(f_q, x0, d0, grad_q)
    assert num_iter == 7
    assert eta == pytest.approx(0.87**7)


def test_wolfe_strong_accepts_known_step():
    num_iter, eta = wolfe(f_q, x0, d0, grad_q, types="strong")
    assert num_iter == 10
    assert eta == pytest.approx(0.87**10)


def test_wolfe_weak_accepts_known_step():
    # weak Wolfe tolerates the overshoot that strong rejects
    num_iter, eta = wolfe(f_q, x0, d0, grad_q, types="weak")
    assert num_iter == 7
    assert eta == pytest.approx(0.87**7)


def test_finite_difference_fallback_matches_analytic():
    # central differences are exact for quadratics, so the accepted
    # steps must match the analytic-gradient runs
    assert armijo(f_q, x0, d0)[1] == pytest.approx(0.87**7)
    assert wolfe(f_q, x0, d0, types="strong")[1] == pytest.approx(0.87**10)
    assert wolfe(f_q, x0, d0, types="weak")[1] == pytest.approx(0.87**7)


def test_torch_gradient_source_matches_analytic():
    torch = pytest.importorskip("torch")
    from mopt.autodiff import torch_gradient

    Qt = torch.as_tensor(Q)
    grad = torch_gradient(lambda xt: xt @ Qt @ xt)
    assert armijo(f_q, x0, d0, grad) == (7, pytest.approx(0.87**7))
    assert wolfe(f_q, x0, d0, grad, types="strong") == (10, pytest.approx(0.87**10))


def test_ascent_direction_rejected():
    with pytest.raises(ValueError, match="descent"):
        armijo(f_q, x0, -d0, grad_q)
    with pytest.raises(ValueError, match="descent"):
        wolfe(f_q, x0, -d0, grad_q)


def test_wolfe_invalid_types():
    with pytest.raises(ValueError, match="strong"):
        wolfe(f_q, x0, d0, grad_q, types="medium")


def test_wolfe_window_skip_raises():
    # eta0=1.2, delta=0.5 hops over the strong acceptance window
    # [0.191, 0.281] (trials 0.6, 0.3, 0.15, ...) and must fail loudly
    with pytest.raises(RuntimeError, match="no acceptable step"):
        wolfe(f_q, x0, d0, grad_q, types="strong", eta=1.2, delta=0.5)


def test_armijo_stationary_point_rejected():
    # at the minimizer the slope is ~0, never < 0
    with pytest.raises(ValueError, match="descent"):
        armijo(f_q, np.zeros(2), d0, grad_q)
