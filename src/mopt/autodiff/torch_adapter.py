"""Exact gradients via PyTorch autodiff, behind a NumPy-facing interface.

PyTorch is an optional dependency (``pip install mopt[torch]``); it is
imported lazily so the rest of mopt works without it. Functions passed to
this module must be written with torch operations — autodiff cannot see
through plain-NumPy code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mopt.nonlinear import NLPProblem


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for mopt.autodiff.torch_adapter; "
            "install it with: pip install mopt[torch]"
        ) from exc
    return torch


def torch_function(f) -> Callable[[np.ndarray], float]:
    """Wrap a torch-written scalar function as a NumPy-facing objective."""

    def f_np(x: np.ndarray) -> float:
        torch = _torch()
        with torch.no_grad():
            return float(f(torch.as_tensor(np.asarray(x, dtype=float))))

    return f_np


def torch_gradient(f) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a torch-written scalar function as a NumPy-facing gradient.

    The returned callable evaluates the exact gradient of ``f`` by reverse-
    mode autodiff: one forward pass builds the computation graph, one
    backward pass applies the chain rule — the full gradient costs roughly
    one extra evaluation of ``f`` regardless of the dimension.

    Parameters
    ----------
    f : callable
        Maps a 1-D float64 torch tensor to a scalar torch tensor, using
        torch operations throughout.

    Returns
    -------
    callable
        ``grad(x: np.ndarray) -> np.ndarray`` with float64 precision.
    """

    def grad(x: np.ndarray) -> np.ndarray:
        torch = _torch()
        xt = torch.tensor(np.asarray(x, dtype=float), requires_grad=True)
        (g,) = torch.autograd.grad(f(xt), xt)
        return g.numpy()

    return grad


def from_torch(f, x0) -> NLPProblem:
    """Build an :class:`~mopt.nonlinear.NLPProblem` from a torch-written ``f``.

    The objective and its exact autodiff gradient are both wired in, so the
    resulting problem never falls back to finite differences.
    """
    return NLPProblem(f=torch_function(f), x0=x0, grad=torch_gradient(f))
