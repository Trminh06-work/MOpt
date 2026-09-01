"""Nonlinear programming problems and solvers.

Problem descriptions, the shared line searches and the differentiation
helpers live here; the solvers are split by what they optimize over —
:mod:`~mopt.nonlinear.unconstrained` and
:mod:`~mopt.nonlinear.constrained`. Everything is re-exported at this
level, so ``from mopt.nonlinear import Newton, ProjectedGradient`` works
regardless of which subpackage a solver lives in.
"""

from mopt.nonlinear.constrained import (
    SQP,
    FrankWolfe,
    KKTResidual,
    LinearOracle,
    PenaltyBarrierMethod,
    PenaltyMethod,
    ProjectedGradient,
    Projection,
    affine_projection,
    barrier_merit,
    constraint_violation,
    damped_bfgs,
    ellipsoid_oracle,
    ellipsoid_projection,
    frank_wolfe_gap,
    halfspace_projection,
    intersection_projection,
    kkt_residual,
    lagrangian,
    penalty_merit,
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
    "SQP",
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
    "PenaltyBarrierMethod",
    "PenaltyMethod",
    "ProjectedGradient",
    "Projection",
    "QuadraticCG",
    "QuasiNewton",
    "QuasiNewtonUpdate",
    "TrustRegion",
    "TrustRegionMethod",
    "affine_projection",
    "armijo",
    "barrier_merit",
    "bfgs",
    "bracketing_wolfe",
    "cauchy",
    "constraint_violation",
    "damped_bfgs",
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
    "penalty_merit",
    "polak_ribiere",
    "projected_gradient_residual",
    "wolfe",
]
