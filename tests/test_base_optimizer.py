import numpy as np
import pytest

from mopt import BaseOptimizer, OptimizeResult


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseOptimizer()


def test_subclass_must_implement_solve():
    class Incomplete(BaseOptimizer):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_solve_works():
    class Constant(BaseOptimizer):
        def solve(self, problem):
            return OptimizeResult(x=np.zeros(2), fun=0.0, success=True)

    result = Constant().solve(problem=None)
    assert result.success
    assert result.fun == 0.0
    assert result.message == ""
    assert result.n_iter == 0
