"""test_matlab - tests the constructors for MATLAB CORA (cora.matlab)

Syntax:   python -m pytest tests/python/test_matlab.py
Outputs:  pass or fail
See also: test_numpy
"""
import numpy as np

from cora import matlab

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


def test_constructors():
    Z = matlab.zonotope(matlab.tensor([1.0, 2.0]), matlab.tensor([[1.0, 0.0], [0.0, 2.0]]))
    assert np.allclose(Z.center, [1, 2]) and np.allclose(Z.generators, [[1, 0], [0, 2]])

    I = matlab.interval(matlab.tensor([0.0, 1.0]), matlab.tensor([2.0, 3.0]))
    assert np.allclose(I.inf, [0, 1]) and np.allclose(I.sup, [2, 3])

    M = matlab.tensor([[0.0, 1.0], [2.0, 3.0]])
    assert np.allclose(matlab.interval_matrix(M, M + 1).sup, M + 1)


def test_reach_on_a_linear_system():
    sys = matlab.linear_sys(matlab.tensor([[-1.0, 0.0], [0.0, -2.0]]))
    X0 = matlab.zonotope(matlab.tensor([1.0, 1.0]), matlab.tensor([[0.1, 0.0], [0.0, 0.1]]))
    R = sys.reach(X0, time_step=0.1, t_final=0.5, taylor_terms=4)
    assert len(R.time_point) == 6 and len(R.time_interval) == 5

def test_reach_with_an_input_and_an_output():
    sys = matlab.linear_sys(matlab.tensor([[-1.0, 0.0], [0.0, -2.0]]), matlab.tensor([[1.0], [0.5]]),
                           matlab.tensor([[1.0, 0.0]]))
    X0 = matlab.zonotope(matlab.tensor([1.0, 1.0]), matlab.tensor([[0.1, 0.0], [0.0, 0.1]]))
    U = matlab.zonotope(matlab.tensor(0.5), matlab.tensor(0.1))  # MATLAB passes scalars
    R = sys.reach(X0, time_step=0.1, t_final=0.5, taylor_terms=4, alg="standard", input_set=U, zonotope_order=20)
    assert len(R.time_interval) == 5 and sys.output_set(R).time_interval[0].center.shape == (1,)


def test_zonotopes_of_a_reach_set():
    sys = matlab.linear_sys(matlab.tensor([[-1.0, 0.0], [0.0, -2.0]]))
    X0 = matlab.zonotope(matlab.tensor([1.0, 1.0]), matlab.tensor([[0.1, 0.0], [0.0, 0.1]]))
    sets = sys.reach(X0, time_step=0.1, t_final=0.5, taylor_terms=4).time_interval
    centers, generators, counts = matlab.zonotopes(sets)
    assert centers.shape == (5, 2) and generators.shape == (2, counts.sum())
    first = int(counts[0])
    assert np.allclose(centers[0], sets[0].center) and np.allclose(generators[:, :first], sets[0].generators)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
