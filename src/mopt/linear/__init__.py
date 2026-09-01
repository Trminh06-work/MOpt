"""Linear programming problems and solvers.

:class:`LPProblem` states a program in the standard form
``min c'x  s.t.  A_ub x <= b_ub, A_eq x == b_eq, x >= 0``; solvers convert
to whatever form they need. :class:`Simplex` is the only solver so far.
"""

from mopt.linear.problem import LPProblem
from mopt.linear.simplex import Simplex

__all__ = ["LPProblem", "Simplex"]
