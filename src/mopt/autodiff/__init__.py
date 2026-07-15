"""Automatic differentiation helpers (optional backends)."""

from mopt.autodiff.torch_adapter import from_torch, torch_function, torch_gradient

__all__ = ["from_torch", "torch_function", "torch_gradient"]
