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


def bracketing_wolfe(
    f: Callable,
    x: np.ndarray,
    d: np.ndarray,
    grad_f: Callable | None = None,
    eta: float = 1.0,
    eta_max: float = 1e10,
    gamma: float = 1e-4,
    sigma: float = 0.9,
    max_iter: int = 100,
) -> tuple[int, float]:
    """Strong Wolfe line search that brackets, then zooms.

    Unlike :func:`armijo` and :func:`wolfe`, which only ever *shrink* the
    trial step, this search grows it as well: it doubles ``eta`` until it
    straddles an acceptable step, then bisects the resulting bracket. That
    matters whenever the acceptable window lies above the initial trial —
    a shrink-only search can never reach it and fails outright, however
    fine its shrink factor.

    Quasi-Newton directions need exactly this. Their natural step is
    :math:`\\eta = 1`, but only once ``H`` approximates the inverse
    Hessian; before then, and on curved valleys, the window routinely sits
    above 1. Nocedal & Wright, Algorithms 3.5 and 3.6.

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
        central finite differences of ``f`` are used.
    eta : float
        First trial step. Kept at 1.0 for Newton and quasi-Newton
        directions, where the unit step is the asymptotically right one.
    eta_max : float
        Ceiling on the bracketing phase.
    gamma : float
        Sufficient-decrease constant, in (0, ``sigma``).
    sigma : float
        Curvature constant, in (``gamma``, 1). The 0.9 default is the
        value Nocedal & Wright prescribe for Newton and quasi-Newton
        directions; the tighter values used for conjugate gradients demand
        a near-exact minimizer of every line.
    max_iter : int
        Maximum trial steps across both phases.

    Returns
    -------
    num_iter : int
        Number of trial steps evaluated.
    eta : float
        The accepted step size.

    Raises
    ------
    ValueError
        If the constants are out of range, or ``d`` is not a descent
        direction at ``x``.
    RuntimeError
        If no acceptable step is found within ``max_iter`` trials.
    """
    if not 0 < gamma < sigma < 1:
        raise ValueError("need 0 < gamma < sigma < 1.")
    if grad_f is None:
        grad_f = partial(finite_difference_gradient, f)

    phi_0 = f(x)
    slope_0 = np.dot(grad_f(x), d)
    if slope_0 >= 0:
        raise ValueError("d is not a descent direction: grad_f(x) @ d >= 0")

    def phi(step: float) -> float:
        return f(x + step * d)

    def slope(step: float) -> float:
        return float(np.dot(grad_f(x + step * d), d))

    def zoom(lo: float, hi: float, phi_lo: float, budget: int) -> tuple[int, float]:
        # invariant: [lo, hi] contains a step meeting both conditions, with
        # lo the better endpoint; bisect until one is found
        for used in range(1, budget + 1):
            mid = 0.5 * (lo + hi)
            phi_mid = phi(mid)
            if phi_mid > phi_0 + gamma * mid * slope_0 or phi_mid >= phi_lo:
                hi = mid
            else:
                slope_mid = slope(mid)
                if abs(slope_mid) <= -sigma * slope_0:
                    return used, mid
                if slope_mid * (hi - lo) >= 0:
                    hi = lo
                lo, phi_lo = mid, phi_mid
        raise RuntimeError(
            f"bracketing_wolfe: no acceptable step within {max_iter} trials"
        )

    eta_prev, phi_prev = 0.0, phi_0
    for num_iter in range(1, max_iter + 1):
        phi_eta = phi(eta)
        if phi_eta > phi_0 + gamma * eta * slope_0 or (
            num_iter > 1 and phi_eta >= phi_prev
        ):
            used, step = zoom(eta_prev, eta, phi_prev, max_iter - num_iter)
            return num_iter + used, step
        slope_eta = slope(eta)
        if abs(slope_eta) <= -sigma * slope_0:
            return num_iter, eta
        if slope_eta >= 0:
            used, step = zoom(eta, eta_prev, phi_eta, max_iter - num_iter)
            return num_iter + used, step
        eta_prev, phi_prev = eta, phi_eta
        eta = min(2.0 * eta, eta_max)
    raise RuntimeError(
        f"bracketing_wolfe: no acceptable step within {max_iter} trials"
    )
