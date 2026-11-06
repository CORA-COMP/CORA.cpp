"""test_matlab - tests the constructors for MATLAB CORA (cora.matlab)

Syntax:   python -m unittest tests.python.test_matlab
Outputs:  pass or fail
See also: test_numpy

Authors:       Claude Sonnet 5.5, Tobias Ladner
Written:       30-September-2026
Last update:   02-November-2026
Reviewed:      ---
"""
import unittest

import numpy as np
from cora import matlab

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

class TestMatlab(unittest.TestCase):

    def test_constructors(self):
        Z = matlab.zonotope(matlab.tensor([1.0, 2.0]), matlab.tensor([[1.0, 0.0], [0.0, 2.0]]))
        self.assertTrue(np.allclose(Z.center, [1, 2]) and np.allclose(Z.generators, [[1, 0], [0, 2]]))

        I = matlab.interval(matlab.tensor([0.0, 1.0]), matlab.tensor([2.0, 3.0]))
        self.assertTrue(np.allclose(I.inf, [0, 1]) and np.allclose(I.sup, [2, 3]))

        M = matlab.tensor([[0.0, 1.0], [2.0, 3.0]])
        self.assertTrue(np.allclose(matlab.intervalMatrix(M, M + 1).sup, M + 1))

    def test_reach_on_a_linear_system(self):
        sys = matlab.linearSys(matlab.tensor([[-1.0, 0.0], [0.0, -2.0]]))
        X0 = matlab.zonotope(matlab.tensor([1.0, 1.0]), matlab.tensor([[0.1, 0.0], [0.0, 0.1]]))
        R = sys.reach(X0, 0.1, 0.5, 4)
        self.assertEqual(len(R.timePoint), 6)
        self.assertEqual(len(R.timeInt), 5)

    def test_reach_with_an_input_and_an_output(self):
        sys = matlab.linearSys(matlab.tensor([[-1.0, 0.0], [0.0, -2.0]]), matlab.tensor([[1.0], [0.5]]),
                               matlab.tensor([[1.0, 0.0]]))
        X0 = matlab.zonotope(matlab.tensor([1.0, 1.0]), matlab.tensor([[0.1, 0.0], [0.0, 0.1]]))
        U = matlab.zonotope(matlab.tensor(0.5), matlab.tensor(0.1))  # MATLAB passes scalars
        R = sys.reach(X0, 0.1, 0.5, 4, "standard", U, 20)
        self.assertEqual(len(R.timeInt), 5)
        self.assertEqual(sys.outputSet(R).timeInt[0].center.shape, (1,))

    def test_zonotopes_of_a_reach_set(self):
        sys = matlab.linearSys(matlab.tensor([[-1.0, 0.0], [0.0, -2.0]]))
        X0 = matlab.zonotope(matlab.tensor([1.0, 1.0]), matlab.tensor([[0.1, 0.0], [0.0, 0.1]]))
        sets = sys.reach(X0, 0.1, 0.5, 4).timeInt
        centers, generators, counts = matlab.zonotopes(sets)
        self.assertEqual(centers.shape, (5, 2))
        self.assertEqual(generators.shape, (2, counts.sum()))
        first = int(counts[0])
        self.assertTrue(
            np.allclose(centers[0], sets[0].center) and
            np.allclose(generators[:, :first], sets[0].generators)
        )

# ----------------------------------------  END OF CODE  ----------------------------------------- #
