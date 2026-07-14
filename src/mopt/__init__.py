"""MOpt: mathematical optimization solvers.

Implements solvers for linear programming (Simplex, ...), nonlinear
programming (line search, Newton's method, ...), and other classes of
optimization problems.
"""

from mopt.base_classes import BaseOptimizer, OptimizeResult

__version__ = "0.0.1"

__all__ = ["BaseOptimizer", "OptimizeResult"]
