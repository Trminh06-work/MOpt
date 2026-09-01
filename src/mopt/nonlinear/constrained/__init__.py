"""Solvers and optimality tests for constrained nonlinear programs.

Every solver here takes a :class:`~mopt.nonlinear.ConstrainedNLPProblem`
and keeps each iterate inside the feasible set. The feasible set reaches
the solvers as an *oracle* — a projection or a linear minimizer — so no
algebraic description of the constraints is needed to optimize.

:mod:`~mopt.nonlinear.constrained.optimality` does need one, and works
from the explicit ``ineq``/``eq`` functions instead, since Lagrange
multipliers are attached to individual constraints.
"""

from mopt.nonlinear.constrained.optimality import (
    KKTResidual,
    frank_wolfe_gap,
    kkt_residual,
    lagrangian,
    projected_gradient_residual,
)
from mopt.nonlinear.constrained.penalty import (
    PenaltyBarrierMethod,
    PenaltyMethod,
    barrier_merit,
    constraint_violation,
    penalty_merit,
)
from mopt.nonlinear.constrained.projected_gradient import (
    FrankWolfe,
    LinearOracle,
    ProjectedGradient,
    Projection,
    affine_projection,
    ellipsoid_oracle,
    ellipsoid_projection,
    halfspace_projection,
    intersection_projection,
)
from mopt.nonlinear.constrained.sqp import SQP, damped_bfgs

__all__ = [
    "SQP",
    "FrankWolfe",
    "KKTResidual",
    "LinearOracle",
    "PenaltyBarrierMethod",
    "PenaltyMethod",
    "ProjectedGradient",
    "Projection",
    "affine_projection",
    "barrier_merit",
    "constraint_violation",
    "damped_bfgs",
    "ellipsoid_oracle",
    "ellipsoid_projection",
    "frank_wolfe_gap",
    "halfspace_projection",
    "intersection_projection",
    "kkt_residual",
    "lagrangian",
    "penalty_merit",
    "projected_gradient_residual",
]
