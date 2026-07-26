"""Nonlinear programming problems and solvers."""

from mopt.nonlinear.conjugate_grad_method import (
    ArmijoModifiedCG,
    BetaRule,
    ConjugateGradient,
    QuadraticCG,
    fletcher_reeves,
    polak_ribiere,
)
from mopt.nonlinear.finite_diff import finite_difference_gradient
from mopt.nonlinear.gradient_descent import GradientDescent
from mopt.nonlinear.line_search import LineSearch, armijo, wolfe
from mopt.nonlinear.newton import Newton
from mopt.nonlinear.problem import NLPProblem
from mopt.nonlinear.trust_region import TrustRegion, TrustRegionMethod, cauchy, dogleg

__all__ = [
    "ArmijoModifiedCG",
    "BetaRule",
    "ConjugateGradient",
    "GradientDescent",
    "LineSearch",
    "NLPProblem",
    "Newton",
    "QuadraticCG",
    "TrustRegion",
    "TrustRegionMethod",
    "armijo",
    "cauchy",
    "dogleg",
    "finite_difference_gradient",
    "fletcher_reeves",
    "polak_ribiere",
    "wolfe",
]
