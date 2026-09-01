"""Exact gradients by automatic differentiation, via optional backends.

Wraps an objective written with torch operations so the NumPy-facing
solvers can use it: :func:`torch_gradient` for the gradient alone,
:func:`from_torch` for a ready :class:`~mopt.nonlinear.NLPProblem`.

PyTorch is an optional extra (``pip install mopt[torch]``) and is imported
lazily, so this package imports cleanly without it. Autodiff only traces
torch operations — a plain-NumPy objective cannot be differentiated this
way, and falls back to finite differences.
"""

from mopt.autodiff.torch_adapter import from_torch, torch_function, torch_gradient

__all__ = ["from_torch", "torch_function", "torch_gradient"]
