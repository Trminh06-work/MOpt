# Examples

Examples that use MOpt the way an installed user would
(`from mopt.nonlinear import ...`), rather than carrying their own copy of
the algorithms.

| Notebook | Covers |
|---|---|
| `line_search.ipynb` | `armijo`, `wolfe`, `bracketing_wolfe`, and plugging one into a solver |
| `linear_programming.ipynb` | `Simplex`: standard form, infeasible and unbounded, degeneracy, cycling |
| `trust_region.ipynb` | `TrustRegion` with `cauchy` and `dogleg`, across starting radii |
| `quasi_newton_method.ipynb` | `QuasiNewton` with `dfp` and `bfgs`, and the `H0` seed |
| `conjugate_grad_method.ipynb` | `QuadraticCG`, `ConjugateGradient`, `ArmijoModifiedCG` |
| `projected_grad_method.ipynb` | `ProjectedGradient` and `FrankWolfe`, projections and oracles |
| `penalty_barrier_method.ipynb` | `PenaltyMethod`, `PenaltyBarrierMethod`, pluggable inner solvers |
| `sequential_quadratic_programming.ipynb` | `SQP`, KKT certificates, `damped_bfgs` |

Every result is checked against SciPy or a known optimum. The notebooks are
stored with their outputs, so they can be read without running them.

## Running them

```bash
pip install -e . --group dev     # mopt, pytest, scipy
pip install pandas jupyter       # the notebooks tabulate results
```
