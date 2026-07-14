"""Linear programming problem description."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LPProblem:
    """A linear program.

    Describes the problem

    .. math::

        \\min_x \\; c^T x
        \\quad \\text{s.t.} \\quad
        A_{ub} x \\le b_{ub}, \\;
        A_{eq} x = b_{eq}, \\;
        x \\ge 0

    Attributes
    ----------
    c : np.ndarray, shape (n,)
        Objective coefficients (minimized).
    A_ub, b_ub : np.ndarray or None
        Inequality constraints ``A_ub @ x <= b_ub``, shapes (m_ub, n) and (m_ub,).
    A_eq, b_eq : np.ndarray or None
        Equality constraints ``A_eq @ x == b_eq``, shapes (m_eq, n) and (m_eq,).
    """

    c: np.ndarray
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None

    def __post_init__(self):
        self.c = np.atleast_1d(np.asarray(self.c, dtype=float))
        if self.c.ndim != 1:
            raise ValueError("c must be a 1-D array.")
        n = self.c.size
        for name in ("ub", "eq"):
            A = getattr(self, f"A_{name}")
            b = getattr(self, f"b_{name}")
            if (A is None) != (b is None):
                raise ValueError(f"A_{name} and b_{name} must be given together.")
            if A is None:
                continue
            A = np.atleast_2d(np.asarray(A, dtype=float))
            b = np.atleast_1d(np.asarray(b, dtype=float))
            if A.shape != (b.size, n):
                raise ValueError(
                    f"A_{name} has shape {A.shape}, expected ({b.size}, {n})."
                )
            setattr(self, f"A_{name}", A)
            setattr(self, f"b_{name}", b)
