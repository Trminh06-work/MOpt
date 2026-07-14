"""Base classes shared by every MOpt optimizer.

All solvers follow one contract: a Problem object describes *what* to
optimize, and each optimizer implements ``solve(problem) -> OptimizeResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OptimizeResult:
    """Outcome of an optimization run, returned by every solver.

    Attributes
    ----------
    x : np.ndarray or None
        The solution found, or None when the run failed (e.g. the problem
        is infeasible or unbounded).
    fun : float or None
        Objective value at ``x``, or None when there is no solution.
    success : bool
        Whether the solver terminated with a valid solution.
    message : str
        Human-readable description of the termination cause.
    n_iter : int
        Number of iterations performed.
    """

    x: np.ndarray | None
    fun: float | None
    success: bool
    message: str = ""
    n_iter: int = 0


class BaseOptimizer(ABC):
    """Abstract base class that all MOpt optimizers must inherit.

    Subclasses implement :meth:`solve`; they must not mutate the problem
    they are given, and must report failures (infeasible, unbounded,
    iteration limit, ...) via ``OptimizeResult.success`` and
    ``OptimizeResult.message`` rather than by raising.
    """

    @abstractmethod
    def solve(self, problem: Any) -> OptimizeResult:
        """Solve the given problem.

        Parameters
        ----------
        problem : Any
            A problem description understood by this solver. Concrete
            Problem classes are introduced alongside the first solver for
            each problem class.

        Returns
        -------
        OptimizeResult
            The solution and termination details.
        """
