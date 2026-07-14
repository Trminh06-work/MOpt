import numpy as np
import pytest
from scipy.optimize import linprog

from mopt.linear import LPProblem, Simplex


def _assert_feasible(problem, x, tol=1e-8):
    assert (x >= -tol).all()
    if problem.A_ub is not None:
        assert (problem.A_ub @ x <= problem.b_ub + tol).all()
    if problem.A_eq is not None:
        np.testing.assert_allclose(problem.A_eq @ x, problem.b_eq, atol=tol)


def test_dantzig_example():
    # max 3x + 5y s.t. x <= 4, 2y <= 12, 3x + 2y <= 18 -> optimum (2, 6)
    problem = LPProblem(
        c=[-3, -5],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
    )
    result = Simplex().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [2, 6], atol=1e-8)
    assert result.fun == pytest.approx(-36.0)
    assert result.n_iter >= 1


def test_equality_constraint():
    # min 2x + 3y s.t. x + y = 10, x <= 6 -> x = 6, y = 4
    problem = LPProblem(c=[2, 3], A_ub=[[1, 0]], b_ub=[6], A_eq=[[1, 1]], b_eq=[10])
    result = Simplex().solve(problem)
    assert result.success
    np.testing.assert_allclose(result.x, [6, 4], atol=1e-8)
    assert result.fun == pytest.approx(24.0)


def test_negative_rhs():
    # min x s.t. x >= 2, written as -x <= -2
    problem = LPProblem(c=[1], A_ub=[[-1]], b_ub=[-2])
    result = Simplex().solve(problem)
    assert result.success
    assert result.fun == pytest.approx(2.0)


def test_infeasible():
    # x <= 1 and x >= 2
    problem = LPProblem(c=[1], A_ub=[[1], [-1]], b_ub=[1, -2])
    result = Simplex().solve(problem)
    assert not result.success
    assert "infeasible" in result.message.lower()
    assert result.x is None


def test_unbounded():
    # min -x with only y bounded
    problem = LPProblem(c=[-1, 0], A_ub=[[0, 1]], b_ub=[1])
    result = Simplex().solve(problem)
    assert not result.success
    assert "unbounded" in result.message.lower()


def test_unconstrained():
    # x >= 0 alone: minimum of x1 + x2 is at the origin
    problem = LPProblem(c=[1, 1])
    result = Simplex().solve(problem)
    assert result.success
    assert result.fun == pytest.approx(0.0)


def test_degenerate_vertex():
    # (1, 1) is over-determined: all three constraints active there
    problem = LPProblem(c=[-1, -1], A_ub=[[1, 0], [0, 1], [1, 1]], b_ub=[1, 1, 2])
    result = Simplex().solve(problem)
    assert result.success
    assert result.fun == pytest.approx(-2.0)


def test_redundant_equality():
    # Duplicated equality row: phase 1 must drop the redundant row
    problem = LPProblem(c=[1, 2], A_eq=[[1, 1], [1, 1]], b_eq=[5, 5])
    result = Simplex().solve(problem)
    assert result.success
    assert result.fun == pytest.approx(5.0)


def test_beale_cycling_example():
    # Classic example that cycles without an anti-cycling rule
    problem = LPProblem(
        c=[-0.75, 150, -0.02, 6],
        A_ub=[
            [0.25, -60, -0.04, 9],
            [0.5, -90, -0.02, 3],
            [0, 0, 1, 0],
        ],
        b_ub=[0, 0, 1],
    )
    result = Simplex().solve(problem)
    reference = linprog(problem.c, A_ub=problem.A_ub, b_ub=problem.b_ub)
    assert result.success and reference.success
    assert result.fun == pytest.approx(reference.fun)
    _assert_feasible(problem, result.x)


def test_random_battery_vs_scipy():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(1, 9))
        # b >= 0 keeps x = 0 feasible; x <= 10 rows keep the LP bounded
        problem = LPProblem(
            c=rng.normal(size=n),
            A_ub=np.vstack([rng.normal(size=(m, n)), np.eye(n)]),
            b_ub=np.concatenate([rng.uniform(0.5, 5.0, size=m), np.full(n, 10.0)]),
        )
        result = Simplex().solve(problem)
        reference = linprog(problem.c, A_ub=problem.A_ub, b_ub=problem.b_ub)
        assert result.success and reference.success
        assert result.fun == pytest.approx(reference.fun, abs=1e-7)
        _assert_feasible(problem, result.x)


def test_random_battery_with_equalities_vs_scipy():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(3, 7))
        k = int(rng.integers(1, 3))
        # b_eq = A_eq @ x0 with x0 >= 0 guarantees feasibility
        A_eq = rng.normal(size=(k, n))
        x0 = rng.uniform(0.0, 3.0, size=n)
        problem = LPProblem(
            c=rng.normal(size=n),
            A_ub=np.eye(n),
            b_ub=np.full(n, 10.0),
            A_eq=A_eq,
            b_eq=A_eq @ x0,
        )
        result = Simplex().solve(problem)
        reference = linprog(
            problem.c, A_ub=problem.A_ub, b_ub=problem.b_ub,
            A_eq=problem.A_eq, b_eq=problem.b_eq,
        )
        assert result.success and reference.success
        assert result.fun == pytest.approx(reference.fun, abs=1e-7)
        _assert_feasible(problem, result.x)


def test_problem_validation():
    with pytest.raises(ValueError):
        LPProblem(c=[1, 2], A_ub=[[1, 2]])  # A_ub without b_ub
    with pytest.raises(ValueError):
        LPProblem(c=[1, 2], A_ub=[[1, 2, 3]], b_ub=[1])  # wrong width
