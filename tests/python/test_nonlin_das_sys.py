"""test_nonlin_das_sys - NonlinDASys in Python: the Newton solvers, reach and simulate.

x' = -x + y, 0 = y - x/2 - u is x' = -x/2 + u with y = x/2 + u; x' = y, 0 = y - x^2 is x' = x^2.

CORA authors: Matthias Althoff, Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@nonlinDASys/nonlinDASys.m
# CORA: contDynamics/@nonlinDASys/consistentInitialState.m
# CORA: contDynamics/@nonlinDASys/initReach.m
# CORA: contDynamics/@nonlinDASys/linReach.m
# CORA: contDynamics/@nonlinDASys/post.m
# CORA: contDynamics/@nonlinDASys/simulate.m
# CORA: contDynamics/@nonlinDASys/steadyState.m
# CORA: contDynamics/@contDynamics/reach.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def linear_system():
    """x' = -x + y, 0 = y - x/2 - u."""
    return cora.NonlinDASys(lambda x, y, u: [-1 * x[0] + y[0]],
                            lambda x, y, u: [y[0] - 0.5 * x[0] - u[0]], 1, inputs=1, constraints=1)


def square_system():
    """x' = y, 0 = y - x^2."""
    return cora.NonlinDASys(lambda x, y, u: [y[0]], lambda x, y, u: [y[0] - x[0] ** 2], 1)


# ============================================  MAIN  ============================================ #

class NonlinDASys(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_newton_solvers(self):
        sys = square_system()
        self.assertEqual((sys.dim, sys.input_dim, sys.constraint_dim), (1, 0, 1))
        self.assertAlmostEqual(sys.consistent_initial_state([1.5], [0.0])[0], 2.25)
        x, y = linear_system().steady_state([0.0], [0.0], [0.5])
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(y[0], 1.0)

    def test_reach_encloses_the_closed_form(self):
        # x' = -x/2 + u with u = 0.2: x(t) = e^{-t/2} + 0.4 (1 - e^{-t/2}) from x(0) = 1
        X0 = cora.Zonotope(np.array([1.0]), np.array([[0.1]]))
        U = cora.Zonotope(np.array([0.0]), np.array([[0.1]]))
        # the constant input 0.2 and the input set [-0.1, 0.1] around it
        R = linear_system().reach(initial_set=X0, input_set=U, u_trans=np.array([[0.2]]),
                                  y0_guess=np.array([0.0]), t_final=1.0, time_step=0.05,
                                  taylor_terms=6)
        self.assertEqual(len(R.time_interval), 20)
        for k in range(1, 21):
            t = 0.05 * k
            exact = np.exp(-t / 2) + 0.4 * (1 - np.exp(-t / 2))
            self.assertTrue(R.time_point[k].contains(np.array([exact])), k)

    def test_simulate_follows_the_reduced_system(self):
        x, y = linear_system().simulate([1.0], [0.0], 1.0, 0.1, u=[0.2])
        exact = np.exp(-0.5) + 0.4 * (1 - np.exp(-0.5))
        self.assertEqual(x.shape, (1, 11))
        self.assertAlmostEqual(x[0, -1], exact, places=6)
        self.assertAlmostEqual(y[0, -1], 0.5 * exact + 0.2, places=6)

    def test_wrong_input_is_refused(self):
        with self.assertRaises(cora.errors.WrongValue):
            square_system().reach(initial_set=cora.Zonotope(np.array([0.4]), np.array([[0.2]])),
                                  t_final=1.0, time_step=0.01)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
