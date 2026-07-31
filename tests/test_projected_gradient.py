from functools import partial

import numpy as np
import pytest

from mopt import BaseOptimizer
from mopt.nonlinear import (
    ConstrainedNLPProblem,
    FrankWolfe,
    ProjectedGradient,
    affine_projection,
    armijo,
    ellipsoid_oracle,
    ellipsoid_projection,
    halfspace_projection,
    intersection_projection,
)

# the lab problem: min x'Qx + b'x  s.t.  c'x = 23,  x'Mx <= 167
Q = np.array([
    [7.0, -0.2, 0.1, -0.3, 0.0],
    [-0.2, 6.9, 0.4, 0.0, 0.1],
    [0.1, 0.4, 6.6, 0.3, 0.0],
    [-0.3, 0.0, 0.3, 6.7, 0.1],
    [0.0, 0.1, 0.0, 0.1, 6.9],
])
M = np.array([
    [5.0, 0.4, -0.1, 0.1, 0.5],
    [0.4, 6.0, -0.5, -2.5, 1.8],
    [-0.1, -0.5, 4.7, 0.1, -1.5],
    [0.1, -2.5, 0.1, 4.5, -0.3],
    [0.5, 1.8, -1.5, -0.3, 3.7],
])
b = np.array([4.0, -3.0, -3.0, 2.0, 2.0])
c = np.array([-4.0, 1.0, 2.0, 4.0, -5.0])
S_ELL, S_EQ = 167.0, 23.0

# equality-constrained optimum, in closed form from the KKT system
_KKT = np.block([[2 * Q, c[:, None]], [c[None, :], np.zeros((1, 1))]])
X_STAR = np.linalg.solve(_KKT, np.concatenate([-b, [S_EQ]]))[:5]
F_STAR = float(X_STAR @ Q @ X_STAR + b @ X_STAR)

# unconstrained optimum, for the cases where no constraint binds
_X_UNC = np.linalg.solve(2 * Q, -b)
F_UNCONSTRAINED = float(_X_UNC @ Q @ _X_UNC + b @ _X_UNC)


def f(x):
    return x @ Q @ x + b @ x


def grad(x):
    return 2.0 * Q @ x + b


def constrained_problem(x0=None, **oracles):
    return ConstrainedNLPProblem(
        f=f, x0=np.zeros(5) if x0 is None else x0, grad=grad, **oracles
    )


# --- projectors ---------------------------------------------------------

def test_halfspace_projection_leaves_feasible_points_alone():
    project = halfspace_projection(c, S_EQ)
    inside = np.zeros(5)  # c @ 0 = 0 <= 23
    np.testing.assert_allclose(project(inside), inside)


def test_halfspace_projection_lands_on_the_boundary():
    project = halfspace_projection(c, S_EQ)
    p = project(10.0 * c)  # far outside
    assert c @ p == pytest.approx(S_EQ)


def test_halfspace_projection_rejects_zero_normal():
    with pytest.raises(ValueError, match="nonzero"):
        halfspace_projection(np.zeros(5), 1.0)


def test_affine_projection_satisfies_the_equality():
    project = affine_projection(c[None, :], [S_EQ])
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = project(rng.uniform(-5, 5, 5))
        assert c @ p == pytest.approx(S_EQ)


def test_affine_projection_is_the_nearest_such_point():
    # for a single hyperplane the correction is collinear with the normal
    # (either sense, depending on which side of it y lies)
    project = affine_projection(c[None, :], [S_EQ])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    residual = y - project(y)
    assert abs(residual @ c) == pytest.approx(
        np.linalg.norm(residual) * np.linalg.norm(c)
    )


def test_affine_projection_checks_shapes():
    with pytest.raises(ValueError, match="rows"):
        affine_projection(np.ones((2, 5)), [1.0])


def test_ellipsoid_projection_inactive_and_active():
    project = ellipsoid_projection(M, S_ELL)
    inside = np.zeros(5)
    np.testing.assert_allclose(project(inside), inside)
    outside = np.full(5, 5.0)  # x'Mx = 497.5 > 167
    p = project(outside)
    assert p @ M @ p == pytest.approx(S_ELL, rel=1e-9)


def test_ellipsoid_projection_is_idempotent_on_the_boundary():
    # re-projecting a boundary point must not move it, and must not raise:
    # the two ways of evaluating the constraint straddle zero by rounding
    project = ellipsoid_projection(M, S_ELL)
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = project(rng.uniform(-8, 8, 5))
        np.testing.assert_allclose(project(p), p, atol=1e-12)


def test_ellipsoid_projection_satisfies_variational_inequality():
    # (y - P(y)) @ (z - P(y)) <= 0 for every feasible z: this is the
    # property ProjectedGradient's descent guarantee is built on
    project = ellipsoid_projection(M, S_ELL)
    rng = np.random.default_rng(2)
    for _ in range(50):
        y = rng.uniform(-8, 8, 5)
        p = project(y)
        for _ in range(5):
            z = project(rng.uniform(-8, 8, 5))
            assert (y - p) @ (z - p) <= 1e-9


def test_ellipsoid_projection_validates_matrix():
    with pytest.raises(ValueError, match="symmetric"):
        ellipsoid_projection(np.array([[1.0, 2.0], [3.0, 4.0]]), 1.0)
    with pytest.raises(ValueError, match="positive definite"):
        ellipsoid_projection(np.diag([1.0, -1.0]), 1.0)
    with pytest.raises(ValueError, match="s must be positive"):
        ellipsoid_projection(M, 0.0)


def test_intersection_projection_beats_alternating_projections():
    # Dykstra must find the NEAREST point of the intersection; plain
    # alternating projections finds only some feasible point, and differs
    # exactly on the draws where both constraints are active
    parts = [ellipsoid_projection(M, 1.0), halfspace_projection(c, 1.0)]
    dykstra = intersection_projection(parts)
    rng = np.random.default_rng(3)
    improved = 0
    for _ in range(30):
        y = rng.uniform(-3, 3, 5)
        p = dykstra(y)
        assert p @ M @ p <= 1.0 + 1e-8 and c @ p <= 1.0 + 1e-8
        pocs = y.copy()
        for _ in range(2000):
            before = pocs
            for part in parts:
                pocs = part(pocs)
            if np.linalg.norm(pocs - before) <= 1e-14:
                break
        if np.linalg.norm(p - y) < np.linalg.norm(pocs - y) - 1e-6:
            improved += 1
    assert improved > 0


def test_intersection_projection_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        intersection_projection([])


def test_ellipsoid_oracle_is_on_the_boundary_and_minimizes():
    lmo = ellipsoid_oracle(M, S_ELL)
    rng = np.random.default_rng(4)
    project = ellipsoid_projection(M, S_ELL)
    for _ in range(20):
        g = rng.uniform(-5, 5, 5)
        y = lmo(g)
        assert y @ M @ y == pytest.approx(S_ELL, rel=1e-9)
        for _ in range(10):  # no feasible point does better
            assert g @ y <= g @ project(rng.uniform(-8, 8, 5)) + 1e-9
    np.testing.assert_allclose(lmo(np.zeros(5)), np.zeros(5))


# --- solvers ------------------------------------------------------------

def test_projected_gradient_matches_kkt_solution():
    projection = intersection_projection([
        ellipsoid_projection(M, S_ELL), affine_projection(c[None, :], [S_EQ]),
    ])
    solver = ProjectedGradient(alpha=1.0)
    assert isinstance(solver, BaseOptimizer)
    result = solver.solve(constrained_problem(projection=projection))
    assert result.success
    assert result.fun == pytest.approx(F_STAR, abs=1e-6)
    np.testing.assert_allclose(result.x, X_STAR, atol=1e-5)


@pytest.mark.parametrize("alpha", [0.05, 0.25, 1.0])
def test_projected_gradient_is_feasible_for_every_alpha(alpha):
    projection = intersection_projection([
        ellipsoid_projection(M, S_ELL), affine_projection(c[None, :], [S_EQ]),
    ])
    rng = np.random.default_rng(5)
    for x0 in rng.uniform(-3, 3, size=(3, 5)):
        result = ProjectedGradient(alpha=alpha).solve(
            constrained_problem(x0, projection=projection)
        )
        assert result.success
        assert result.fun == pytest.approx(F_STAR, abs=1e-6)
        assert c @ result.x == pytest.approx(S_EQ)
        assert result.x @ M @ result.x <= S_ELL + 1e-8


def test_projected_gradient_smaller_alpha_needs_fewer_iterations():
    # alpha ~ 1/L is the standard choice and is much faster than alpha = 1
    projection = intersection_projection([
        ellipsoid_projection(M, S_ELL), affine_projection(c[None, :], [S_EQ]),
    ])
    inv_L = 1.0 / (2.0 * np.linalg.eigvalsh(Q).max())
    tuned = ProjectedGradient(alpha=inv_L).solve(
        constrained_problem(projection=projection))
    plain = ProjectedGradient(alpha=1.0).solve(
        constrained_problem(projection=projection))
    assert tuned.success and plain.success
    assert tuned.n_iter < plain.n_iter


def test_projected_gradient_requires_projection():
    with pytest.raises(ValueError, match="projection"):
        ProjectedGradient().solve(constrained_problem())


def test_projected_gradient_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        ProjectedGradient(alpha=0.0)


def test_projected_gradient_reports_iteration_limit():
    projection = ellipsoid_projection(M, S_ELL)
    result = ProjectedGradient(alpha=1.0, max_iter=2).solve(
        constrained_problem(projection=projection))
    assert not result.success
    assert "Iteration limit" in result.message
    assert result.n_iter == 2


# the ellipsoid binds at this level, which is the regime Frank-Wolfe is for
S_TIGHT = 1.0


def test_frank_wolfe_matches_projected_gradient():
    # ellipsoid only, where the linear oracle is available in closed form
    result = FrankWolfe(max_iter=5000).solve(constrained_problem(
        lmo=ellipsoid_oracle(M, S_TIGHT),
        projection=ellipsoid_projection(M, S_TIGHT),
    ))
    assert result.success
    assert result.x @ M @ result.x <= S_TIGHT + 1e-8

    reference = ProjectedGradient(alpha=0.05, max_iter=5000).solve(
        constrained_problem(projection=ellipsoid_projection(M, S_TIGHT))
    )
    assert reference.success
    # Frank-Wolfe converges sublinearly, so it reaches a given accuracy in f
    # well before the same accuracy in x
    assert result.fun == pytest.approx(reference.fun, abs=1e-5)


def test_frank_wolfe_interior_optimum_needs_more_shrinks():
    # documented trap: with the constraint slack the solution is interior, the
    # oracle keeps returning far-away boundary points, and the accepted step
    # goes to zero faster than the default 100 backtracks can reach
    problem = constrained_problem(
        lmo=ellipsoid_oracle(M, S_ELL),          # x*'Mx* = 1.22 << 167
        projection=ellipsoid_projection(M, S_ELL),
    )
    default = FrankWolfe(max_iter=5000).solve(problem)
    assert not default.success
    assert "Line search failed" in default.message
    assert default.fun == pytest.approx(F_UNCONSTRAINED, abs=1e-6)  # yet nearly there

    patient = FrankWolfe(line_search=partial(armijo, max_iter=400),
                         max_iter=5000).solve(problem)
    assert patient.success
    assert patient.fun == pytest.approx(F_UNCONSTRAINED, abs=1e-9)


def test_frank_wolfe_requires_lmo():
    with pytest.raises(ValueError, match="lmo"):
        FrankWolfe().solve(constrained_problem())


def test_frank_wolfe_reports_iteration_limit():
    problem = ConstrainedNLPProblem(f=f, x0=np.zeros(5), grad=grad,
                                    lmo=ellipsoid_oracle(M, S_ELL))
    result = FrankWolfe(max_iter=2).solve(problem)
    assert not result.success
    assert "Iteration limit" in result.message


def test_solvers_use_finite_difference_fallback():
    # no analytic gradient: ConstrainedNLPProblem inherits the fallback
    projection = ellipsoid_projection(M, S_ELL)
    problem = ConstrainedNLPProblem(f=f, x0=np.zeros(5), projection=projection,
                                    lmo=ellipsoid_oracle(M, S_ELL))
    exact = ProjectedGradient(alpha=0.05).solve(
        ConstrainedNLPProblem(f=f, x0=np.zeros(5), grad=grad,
                              projection=projection))
    result = ProjectedGradient(alpha=0.05, tol=1e-5).solve(problem)
    assert result.success
    assert result.fun == pytest.approx(exact.fun, abs=1e-5)


def test_constrained_problem_project_without_oracle():
    with pytest.raises(ValueError, match="no projection oracle"):
        constrained_problem().project(np.zeros(5))
