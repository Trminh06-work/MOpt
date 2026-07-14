"""Two-phase primal simplex method on a dense tableau.

The problem is first brought to standard form ``A x = b, x >= 0, b >= 0`` by
adding one slack variable per inequality row and negating rows with negative
right-hand sides. Phase 1 then minimizes the sum of artificial variables to
find a basic feasible solution (a positive optimum proves infeasibility);
phase 2 minimizes the original objective from that starting basis. Pivots
follow Bland's rule (smallest-index entering and leaving variable), which
guarantees termination even on degenerate problems.
"""

from __future__ import annotations

import numpy as np

from mopt.base_classes import BaseOptimizer, OptimizeResult
from mopt.linear.problem import LPProblem


def _pivot(T: np.ndarray, basis: list[int], row: int, col: int) -> None:
    """Pivot the tableau so that ``col`` becomes basic in ``row``."""
    T[row] /= T[row, col]
    factors = T[:, col].copy()
    factors[row] = 0.0
    T -= np.outer(factors, T[row])
    basis[row] = col


class Simplex(BaseOptimizer):
    """Two-phase tableau simplex solver for :class:`LPProblem`.

    Parameters
    ----------
    max_iter : int
        Maximum number of pivots across both phases.
    tol : float
        Numerical tolerance for optimality, ratio, and feasibility tests.
    """

    def __init__(self, max_iter: int = 10_000, tol: float = 1e-9):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem: LPProblem) -> OptimizeResult:
        c, A, b = self._standard_form(problem)
        m, n_total = A.shape

        # Phase 1: minimize the sum of one artificial variable per row,
        # starting from the all-artificial basis. Tableau layout: one row per
        # constraint plus a reduced-cost row; last column is the RHS, and
        # T[-1, -1] holds minus the current objective value.
        T = np.zeros((m + 1, n_total + m + 1))
        T[:m, :n_total] = A
        T[:m, n_total:-1] = np.eye(m)
        T[:m, -1] = b
        T[-1, :n_total] = -A.sum(axis=0)
        T[-1, -1] = -b.sum()
        basis = list(range(n_total, n_total + m))

        status, it1 = self._iterate(T, basis, self.max_iter)
        if status == "maxiter":
            return OptimizeResult(
                x=None, fun=None, success=False,
                message="Iteration limit reached in phase 1.", n_iter=it1,
            )
        if -T[-1, -1] > self.tol * (1.0 + np.abs(b).sum()):
            return OptimizeResult(
                x=None, fun=None, success=False,
                message="Problem is infeasible.", n_iter=it1,
            )

        # Drive leftover artificials out of the basis (they sit at value 0
        # after a feasible phase 1); a row with no usable pivot is redundant.
        drop = []
        for i in range(m):
            if basis[i] >= n_total:
                j = int(np.argmax(np.abs(T[i, :n_total])))
                if abs(T[i, j]) > self.tol:
                    _pivot(T, basis, i, j)
                else:
                    drop.append(i)
        if drop:
            T = np.delete(T, drop, axis=0)
            basis = [bj for i, bj in enumerate(basis) if i not in drop]
        T = np.hstack([T[:, :n_total], T[:, -1:]])

        # Phase 2: restore the original objective, priced out over the basis.
        T[-1, :-1] = c
        T[-1, -1] = 0.0
        for i, j in enumerate(basis):
            if c[j] != 0.0:
                T[-1] -= c[j] * T[i]

        status, it2 = self._iterate(T, basis, self.max_iter - it1)
        n_iter = it1 + it2
        if status == "unbounded":
            return OptimizeResult(
                x=None, fun=None, success=False,
                message="Problem is unbounded.", n_iter=n_iter,
            )
        if status == "maxiter":
            return OptimizeResult(
                x=None, fun=None, success=False,
                message="Iteration limit reached in phase 2.", n_iter=n_iter,
            )

        x_full = np.zeros(n_total)
        x_full[basis] = T[:-1, -1]
        x = x_full[: problem.c.size]
        return OptimizeResult(
            x=x, fun=float(problem.c @ x), success=True,
            message="Optimal solution found.", n_iter=n_iter,
        )

    def _iterate(self, T: np.ndarray, basis: list[int], max_iter: int):
        """Run simplex pivots until optimal, unbounded, or out of budget."""
        m = len(basis)
        for it in range(max_iter):
            negative = np.where(T[-1, :-1] < -self.tol)[0]
            if negative.size == 0:
                return "optimal", it
            col = int(negative[0])  # Bland's rule: smallest index enters
            column = T[:m, col]
            eligible = column > self.tol
            if not eligible.any():
                return "unbounded", it
            ratios = np.full(m, np.inf)
            ratios[eligible] = T[:m, -1][eligible] / column[eligible]
            ties = np.where(ratios <= ratios.min() + self.tol)[0]
            row = int(min(ties, key=lambda i: basis[i]))  # smallest leaves
            _pivot(T, basis, row, col)
        return "maxiter", max_iter

    @staticmethod
    def _standard_form(problem: LPProblem):
        """Return ``(c, A, b)`` for the equivalent ``A x = b, x >= 0, b >= 0``.

        One slack variable is appended per inequality row; rows with a
        negative right-hand side are negated.
        """
        n = problem.c.size
        m_ub = 0 if problem.A_ub is None else problem.A_ub.shape[0]
        rows, rhs = [], []
        if m_ub:
            rows.append(np.hstack([problem.A_ub, np.eye(m_ub)]))
            rhs.append(problem.b_ub)
        if problem.A_eq is not None:
            m_eq = problem.A_eq.shape[0]
            rows.append(np.hstack([problem.A_eq, np.zeros((m_eq, m_ub))]))
            rhs.append(problem.b_eq)
        if rows:
            A = np.vstack(rows)
            b = np.concatenate(rhs).astype(float)
        else:
            A = np.empty((0, n + m_ub))
            b = np.empty(0)
        neg = b < 0
        A[neg] *= -1
        b[neg] *= -1
        c = np.concatenate([problem.c, np.zeros(m_ub)])
        return c, A, b
