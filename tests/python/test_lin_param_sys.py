"""test_lin_param_sys - LinParamSys in Python: reach of a damped oscillator through the bindings.

CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Authors:       Claude Opus 5.5
Written:       09-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@linParamSys/linParamSys.m
# CORA: contDynamics/@linParamSys/initReach.m
# CORA: contDynamics/@linParamSys/post.m
# CORA: contDynamics/@contDynamics/reach.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def oscillator(kind="constant"):
    """x' = A x + B u + c, A in [-1 2; -2 -1] +- 0.1 I."""
    A = cora.MatZonotope(np.array([[-1.0, 2.0], [-2.0, -1.0]]), np.array([[[0.1, 0.0], [0.0, 0.1]]]))
    return cora.LinParamSys(A, np.array([[0.0], [1.0]]), np.array([0.5, 0.0]), kind)


def nominal(t):
    """The state at t from the center of X0 under the center matrix and the center input."""
    M = np.zeros((3, 3))
    M[:2, :2] = [[-1.0, 2.0], [-2.0, -1.0]]
    M[:2, 2] = [0.5, 0.05]
    term, total = np.eye(3), np.eye(3)
    for i in range(1, 40):
        # the Taylor series of e^{M t}, converged long before
        term = term @ (M * t) / i
        total = total + term
    return (total @ np.array([1.0, 0.5, 1.0]))[:2]


X0 = cora.Zonotope(np.array([1.0, 0.5]), np.array([[0.05, 0.0], [0.0, 0.05]]))
U = cora.Zonotope(np.array([0.05]), np.array([[0.25]]))


# ============================================  MAIN  ============================================ #

class LinParamSys(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_reach_holds_the_nominal_solution(self):
        # the nominal solution stays in the time-point sets
        for kind in ("constant", "varying"):
            R = oscillator(kind).reach(initial_set=X0, input_set=U, t_final=1.0, time_step=0.05,
                                       taylor_terms=6, intermediate_terms=3, zonotope_order=20)
            self.assertEqual(len(R.time_interval), 20)
            for k in range(1, 21):
                self.assertTrue(R.time_point[k].contains(nominal(0.05 * k)), k)

    def test_wrong_input_is_refused(self):
        with self.assertRaises(cora.errors.WrongValue):
            oscillator().reach(initial_set=X0, t_final=1.0, time_step=0.05, intermediate_terms=9)
        self.assertEqual(oscillator().type, "constant")
        with self.assertRaises(cora.errors.WrongValue):
            oscillator("varParam")
        self.assertEqual(oscillator().dim, 2)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
