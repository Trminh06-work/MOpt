"""Nonlinear programming problems and solvers.

Problem descriptions, the shared line searches and the differentiation
helpers live here; the solvers are split by what they optimize over —
:mod:`~mopt.nonlinear.unconstrained` and
:mod:`~mopt.nonlinear.constrained`. Everything is re-exported at this
level, so ``from mopt.nonlinear import Newton, ProjectedGradient`` works
regardless of which subpackage a solver lives in.
"""

from mopt.nonlinear.constrained import (
    FrankWolfe,
    KKTResidual,
    LinearOracle,
    ProjectedGradient,
    Projection,
    affine_projection,
    ellipsoid_oracle,
    ellipsoid_projection,
    frank_wolfe_gap,
    halfspace_projection,
    intersection_projection,
    kkt_residual,
    lagrangian,
    projected_gradient_residual,
)
from mopt.nonlinear.finite_diff import (
    finite_difference_gradient,
    finite_difference_jacobian,
)
from mopt.nonlinear.line_search import (
    LineSearch,
    armijo,
    bracketing_wolfe,
    wolfe,
)
from mopt.nonlinear.problem import ConstrainedNLPProblem, NLPProblem
from mopt.nonlinear.unconstrained import (
    ArmijoModifiedCG,
    BetaRule,
    ConjugateGradient,
    GradientDescent,
    Newton,
    QuadraticCG,
    QuasiNewton,
    QuasiNewtonUpdate,
    TrustRegion,
    TrustRegionMethod,
    bfgs,
    cauchy,
    dfp,
    dogleg,
    fletcher_reeves,
    polak_ribiere,
)

__all__ = [
    "ArmijoModifiedCG",
    "BetaRule",
    "ConjugateGradient",
    "ConstrainedNLPProblem",
    "FrankWolfe",
    "GradientDescent",
    "KKTResidual",
    "LineSearch",
    "LinearOracle",
    "NLPProblem",
    "Newton",
    "ProjectedGradient",
    "Projection",
    "QuadraticCG",
    "QuasiNewton",
    "QuasiNewtonUpdate",
    "TrustRegion",
    "TrustRegionMethod",
    "affine_projection",
    "armijo",
    "bfgs",
    "bracketing_wolfe",
    "cauchy",
    "dfp",
    "dogleg",
    "ellipsoid_oracle",
    "ellipsoid_projection",
    "finite_difference_gradient",
    "finite_difference_jacobian",
    "fletcher_reeves",
    "frank_wolfe_gap",
    "halfspace_projection",
    "intersection_projection",
    "kkt_residual",
    "lagrangian",
    "polak_ribiere",
    "projected_gradient_residual",
    "wolfe",
]
