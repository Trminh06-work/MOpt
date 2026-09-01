# Architecture

How MOpt is put together, and why. For what is implemented, see the [README](README.md).

## The contract

Problems describe *what* to optimize. Solvers implement *how*.

```python
result = Simplex().solve(problem)     # -> OptimizeResult
```

Every solver subclasses `BaseOptimizer` and implements one method,
`solve(problem) -> OptimizeResult`. Solvers do not mutate the problem they are
given. Numerical failures — infeasible, unbounded, iteration limit, a line
search that found no step — are reported through `OptimizeResult.success` and
`.message`. Exceptions are reserved for usage errors: a missing Hessian, a
parameter out of range.

## Layout

```
mopt/
├── base_classes.py     BaseOptimizer, OptimizeResult
├── linear/             LP problem and solvers
├── nonlinear/
│   ├── problem.py      NLPProblem, ConstrainedNLPProblem
│   ├── line_search.py  step-size rules, shared by most solvers
│   ├── finite_diff.py  numerical gradients and Jacobians
│   ├── unconstrained/  searches all of R^n
│   └── constrained/    keeps every iterate in the feasible set
└── autodiff/           optional exact gradients via PyTorch
```

One subpackage per problem class. A subpackage appears when its first solver
lands, not before.

## Abstract classes vs protocols

`BaseOptimizer` is an ABC. Solvers are stateful objects, users subclass them,
and a subclass without `solve` should fail at construction.

The pluggable pieces *inside* solvers are `typing.Protocol` instead:
`LineSearch`, `TrustRegionMethod`, `QuasiNewtonUpdate`, `BetaRule`,
`Projection`, `LinearOracle`. They are plain functions, and a Protocol lets a
function, a `functools.partial` of one, or a user's own callable satisfy the
contract without inheriting anything.

```python
GradientDescent(line_search=partial(wolfe, types="weak"))
TrustRegion(method=dogleg)
QuasiNewton(update=bfgs)
```

There is deliberately no `BaseProblem`. Problems are data, not behaviour, and
`LPProblem` (arrays) and `NLPProblem` (callables) share nothing worth
abstracting.

## Where gradients come from

Solvers call `problem.gradient(x)`, never `problem.grad` directly. That method
resolves the source: the analytic `grad` when supplied, central finite
differences of `f` otherwise. Solvers do not care which.

Exact autodiff gradients plug in as an ordinary callable:

```python
NLPProblem(f=f_numpy, x0=x0, grad=torch_gradient(f_torch))
```

This cannot be automatic. Reverse-mode autodiff only traces torch operations,
and the `f` a solver receives is a NumPy callable.

## Feasible sets

Constrained solvers reach the feasible set through an *oracle* — a projection
(`ProjectedGradient`) or a linear minimizer (`FrankWolfe`). No algebraic
description of the constraints is needed to optimize.

`ConstrainedNLPProblem` also carries optional algebraic `ineq` and `eq`. Those
are required only where individual constraints must be named: KKT multipliers
(`kkt_residual`), and the merit functions of `PenaltyMethod`,
`PenaltyBarrierMethod` and `SQP`.

## Dependencies

NumPy is the only runtime dependency. PyTorch is an optional extra
(`pip install mopt[torch]`), imported lazily inside `mopt.autodiff` so the rest
of the package works without it. SciPy is test-only, used to cross-check
solvers against a trusted implementation.

## labs/

`labs/` holds runnable examples that use the package the way an installed user
would. Not part of the package.

