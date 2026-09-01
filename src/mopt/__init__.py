"""MOpt: mathematical optimization solvers.

Implements solvers for linear programming (Simplex, ...), nonlinear
programming (line search, Newton's method, ...), and other classes of
optimization problems.
"""

from importlib.metadata import PackageNotFoundError, version

from mopt.base_classes import BaseOptimizer, OptimizeResult

try:
    # hatch-vcs derives the version from the git tag at build time; read it
    # back from the installed metadata so there is only one source of truth
    __version__ = version("mopt")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["BaseOptimizer", "OptimizeResult"]
