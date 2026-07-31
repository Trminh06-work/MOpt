"""Solvers for unconstrained nonlinear programs.

Every solver here takes an :class:`~mopt.nonlinear.NLPProblem` and searches
all of :math:`\\mathbb{R}^n`. They are also happy to take a
:class:`~mopt.nonlinear.ConstrainedNLPProblem`, but will ignore its
feasible set — see :mod:`mopt.nonlinear.constrained` for solvers that
respect it.
"""

from mopt.nonlinear.unconstrained.conjugate_grad_method import (
    ArmijoModifiedCG,
    BetaRule,
    ConjugateGradient,
    QuadraticCG,
    fletcher_reeves,
    polak_ribiere,
)
from mopt.nonlinear.unconstrained.gradient_descent import GradientDescent
from mopt.nonlinear.unconstrained.newton import Newton
from mopt.nonlinear.unconstrained.trust_region import (
    TrustRegion,
    TrustRegionMethod,
    cauchy,
    dogleg,
)

__all__ = [
    "ArmijoModifiedCG",
    "BetaRule",
    "ConjugateGradient",
    "GradientDescent",
    "Newton",
    "QuadraticCG",
    "TrustRegion",
    "TrustRegionMethod",
    "cauchy",
    "dogleg",
    "fletcher_reeves",
    "polak_ribiere",
]
