# MOpt

Mathematical optimization solvers in Python, implemented from scratch on NumPy.

> ⚠️ Early development. The API is not yet stable and solvers are being added
> incrementally.

## Installation

```bash
pip install mopt
```

Or the latest development version:

```bash
pip install git+https://github.com/Trminh06-work/MOpt.git
```

PyTorch-backed gradients are an optional extra: `pip install mopt[torch]`.

## Usage

Describe the problem, pick a solver, call `solve`.

```python
import numpy as np
from mopt.nonlinear import NLPProblem, QuasiNewton

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

problem = NLPProblem(f=rosenbrock, x0=[-1.2, 1.0])   # no gradient supplied
result = QuasiNewton().solve(problem)

result.x        # array([1., 1.])
result.fun      # 0.0
result.success  # True
```

Supply `grad` and `hess` when you have them; without `grad`, MOpt falls back to
finite differences.

## Solvers

**Linear programming** — `mopt.linear`

| Solver | Notes |
|---|---|
| `Simplex` | Two-phase tableau, Bland's rule. Detects infeasible and unbounded problems. |

**Unconstrained nonlinear** — `mopt.nonlinear`

| Solver | Notes |
|---|---|
| `GradientDescent` | Steepest descent with a pluggable line search. |
| `Newton` | Damped (line search) or pure unit step. Needs `hess`. |
| `QuasiNewton` | Rank-2 inverse-Hessian updates: `dfp`, `bfgs`. No Hessian needed. |
| `ConjugateGradient` | Nonlinear CG with pluggable beta: `fletcher_reeves`, `polak_ribiere`. |
| `QuadraticCG` | Linear CG for quadratics; exact in at most n steps. |
| `ArmijoModifiedCG` | Nonlinear CG using an Armijo-type step rule. |
| `TrustRegion` | Subproblem solved by `cauchy` or `dogleg`. Needs `hess`. |

**Constrained nonlinear** — `mopt.nonlinear`

| Solver | Feasible set given as | Notes |
|---|---|---|
| `ProjectedGradient` | projection oracle | Every iterate feasible. |
| `FrankWolfe` | linear oracle | Never projects; needs a bounded set. |
| `PenaltyMethod` | `ineq` / `eq` | Quadratic penalty, increasing weight. Iterates infeasible until the limit. |
| `PenaltyBarrierMethod` | `ineq` / `eq` | Log barrier, decreasing weight. Needs a strictly feasible start. |
| `SQP` | `ineq` / `eq` | Damped BFGS model, L1 merit line search, KKT certificate. |

Projections and oracles are built with `affine_projection`,
`halfspace_projection`, `ellipsoid_projection`, `intersection_projection`
(Dykstra) and `ellipsoid_oracle`.

**Supporting pieces**

| | |
|---|---|
| Line searches | `armijo`, `wolfe` (weak/strong), `bracketing_wolfe` |
| Optimality | `kkt_residual`, `lagrangian`, `frank_wolfe_gap`, `projected_gradient_residual` |
| Derivatives | `finite_difference_gradient`, `finite_difference_jacobian`, `mopt.autodiff` (PyTorch) |

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md) for the solver contract, the problem
types, and how the pluggable pieces fit together.

## License

MIT — see [LICENSE](LICENSE).
