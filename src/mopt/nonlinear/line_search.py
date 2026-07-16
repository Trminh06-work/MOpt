"""Line search methods for step-size selection along a descent direction.

Both searches backtrack: the trial step ``eta`` starts optimistic and
shrinks by the factor ``delta`` until an acceptance condition holds. They
operate on NumPy-facing callables, and the gradient source is pluggable —
pass an analytic ``grad_f``, an exact autodiff one built with
:func:`mopt.autodiff.torch_gradient` (or ``problem.gradient`` from an
:class:`~mopt.nonlinear.NLPProblem`), or omit it to fall back to central
finite differences of ``f``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Protocol

import numpy as np

from mopt.nonlinear.finite_diff import finite_difference_gradient


class LineSearch(Protocol):
    """Call contract for line searches pluggable into mopt solvers.

    Any callable with this signature qualifies — the module-level functions
    here, a ``functools.partial`` of one with tunables frozen, or a
    user-defined search. Implementations must return ``(num_iter, eta)``,
    raise ``ValueError`` when ``d`` is not a descent direction, and raise
    ``RuntimeError`` when no acceptable step can be found.
    """

    def __call__(
        self,
        f: Callable,
        x: np.ndarray,
        d: np.ndarray,
        grad_f: Callable | None = None,
    ) -> tuple[int, float]: ...


def armijo(
    f: Callable,
    x: np.ndarray,
    d: np.ndarray,
    grad_f: Callable | None = None,
    eta: float = 1.0,
    delta: float = 0.87,
    gamma: float = 0.14,
    max_iter: int = 100,
) -> tuple[int, float]:
    """Backtracking line search enforcing the Armijo condition.

    Starting from the trial step ``eta``, shrink by the factor ``delta``
    until the sufficient-decrease (Armijo) condition holds:

    .. math::

        f(x + \\eta d) \\le f(x) + \\gamma\\,\\eta\\,\\nabla f(x)^T d

    The right-hand side is the tangent line at ``x`` deflated by ``gamma``;
    since the directional derivative is negative along a descent direction,
    the loop terminates for any smooth ``f``.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float``.
    x : np.ndarray, shape (n,)
        Current point.
    d : np.ndarray, shape (n,)
        Descent direction; must satisfy ``grad_f(x) @ d < 0``.
    grad_f : callable, optional
        Gradient ``grad_f(x) -> np.ndarray`` of shape (n,). When None,
        central finite differences of ``f`` are used. For exact gradients
        of a torch-written objective, pass
        ``mopt.autodiff.torch_gradient(f_torch)``.
    eta : float
        Initial (largest) trial step.
    delta : float
        Shrink factor per rejected trial, in (0, 1).
    gamma : float
        Sufficient-decrease constant, in (0, 1); small values accept
        almost any actual decrease.
    max_iter : int
        Maximum number of shrinks before giving up.

    Returns
    -------
    num_iter : int
        Number of shrinks performed; the accepted step equals
        ``eta * delta**num_iter`` for the initial ``eta``.
    eta : float
        The accepted step size.

    Raises
    ------
    ValueError
        If ``d`` is not a descent direction at ``x``.
    RuntimeError
        If no acceptable step is found within ``max_iter`` shrinks.
    """
    if grad_f is None:
        grad_f = partial(finite_difference_gradient, f)
    fx = f(x)
    slope = np.dot(grad_f(x), d)
    if slope >= 0:
        raise ValueError("d is not a descent direction: grad_f(x) @ d >= 0")
    num_iter = 0
    while f(x + eta * d) > fx + gamma * eta * slope:
        if num_iter >= max_iter:
            raise RuntimeError(
                f"armijo: no acceptable step within {max_iter} shrinks"
            )
        eta *= delta
        num_iter += 1
    return num_iter, eta


def wolfe(
    f: Callable,
    x: np.ndarray,
    d: np.ndarray,
    grad_f: Callable | None = None,
    types: str = "strong",
    eta: float = 1.0,
    delta: float = 0.87,
    gamma: float = 0.14,
    sigma: float = 0.19,
    max_iter: int = 100,
) -> tuple[int, float]:
    """Backtracking line search enforcing the Wolfe conditions.

    A step is accepted when it satisfies both the Armijo sufficient-decrease
    condition (see :func:`armijo`) and a curvature condition on the new
    directional derivative :math:`s(\\eta) = \\nabla f(x + \\eta d)^T d`:

    .. math::

        \\text{weak:} \\quad s(\\eta) \\ge \\sigma\\, s(0)
        \\qquad
        \\text{strong:} \\quad |s(\\eta)| \\le \\sigma\\, |s(0)|

    Armijo rejects steps that are too long; the curvature condition rejects
    steps that are too short (where the slope is still steeply negative).
    Weak Wolfe tolerates overshooting past the valley floor, strong Wolfe
    bounds the new slope from both sides.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float``.
    x : np.ndarray, shape (n,)
        Current point.
    d : np.ndarray, shape (n,)
        Descent direction; must satisfy ``grad_f(x) @ d < 0``.
    grad_f : callable, optional
        Gradient ``grad_f(x) -> np.ndarray`` of shape (n,). When None,
        central finite differences of ``f`` are used. For exact gradients
        of a torch-written objective, pass
        ``mopt.autodiff.torch_gradient(f_torch)``.
    types : {"strong", "weak"}
        Which curvature condition to enforce.
    eta : float
        Initial (largest) trial step.
    delta : float
        Shrink factor per rejected trial, in (0, 1).
    gamma : float
        Sufficient-decrease constant, in (0, 1).
    sigma : float
        Curvature constant, in (``gamma``, 1); smaller values demand a
        flatter slope at the accepted point.
    max_iter : int
        Maximum number of shrinks before giving up.

    Returns
    -------
    num_iter : int
        Number of shrinks performed; the accepted step equals
        ``eta * delta**num_iter`` for the initial ``eta``.
    eta : float
        The accepted step size.

    Raises
    ------
    ValueError
        If ``types`` is invalid, or ``d`` is not a descent direction.
    RuntimeError
        If no acceptable step is found within ``max_iter`` shrinks.

    Notes
    -----
    Because backtracking only ever shrinks the step while the curvature
    condition rejects steps that are too small, a coarse ``delta`` can hop
    over the acceptance window entirely — the search then exhausts
    ``max_iter`` and raises. Bracketing line searches (Nocedal & Wright,
    Algorithms 3.5-3.6) avoid this failure mode; with a fine ``delta``
    (such as the 0.87 default) it is rare in practice.
    """
    if types not in ("strong", "weak"):
        raise ValueError("Type must be `strong` or `weak`")
    if grad_f is None:
        grad_f = partial(finite_difference_gradient, f)
    fx = f(x)
    slope = np.dot(grad_f(x), d)
    if slope >= 0:
        raise ValueError("d is not a descent direction: grad_f(x) @ d >= 0")

    def rejected(step: float) -> bool:
        if f(x + step * d) > fx + gamma * step * slope:
            return True  # insufficient decrease; skip the gradient call
        new_slope = np.dot(grad_f(x + step * d), d)
        if types == "strong":
            return np.abs(new_slope) > sigma * np.abs(slope)
        return new_slope < sigma * slope

    num_iter = 0
    while rejected(eta):
        if num_iter >= max_iter:
            raise RuntimeError(
                f"wolfe ({types}): no acceptable step within {max_iter} shrinks"
            )
        eta *= delta
        num_iter += 1
    return num_iter, eta
