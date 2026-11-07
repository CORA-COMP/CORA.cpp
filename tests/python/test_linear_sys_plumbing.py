"""test_linear_sys_plumbing - LinearSys in Python: equality, repr, generate_random, nonlinear_sys

and the reach algorithms from_start and decomposition.

CORA authors: Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@linearSys/isequal.m
# CORA: contDynamics/@linearSys/eq.m
# CORA: contDynamics/@linearSys/ne.m
# CORA: contDynamics/@linearSys/display.m
# CORA: contDynamics/@linearSys/generateRandom.m
# CORA: contDynamics/@linearSys/nonlinearSys.m
# CORA: contDynamics/@linearSys/private/priv_reach_fromStart.m
# CORA: contDynamics/@linearSys/private/priv_reach_decomp.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def expm(M):
    """The matrix exponential of a diagonalizable M."""
    w, V = np.linalg.eig(M)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real


# a damped oscillator coupled to a decaying state, a set and an input set
A = np.array([[-0.1, 1.0, 0.0], [-1.0, -0.1, 0.0], [0.0, 0.0, -0.5]])
X0 = cora.Zonotope(np.array([1.0, 0.5, -1.0]), np.array([[0.2, 0.0, 0.1], [0.0, 0.1, 0.0], [0.0, 0.0, 0.3]]))
U = cora.Zonotope(np.array([1.0, 2.0, 0.5]), np.array([[0.1, 0.0], [0.0, 0.2], [0.1, 0.1]]))


# ============================================  MAIN  ============================================ #

# the plumbing of the class: equality, text, random systems and the conversion
class Plumbing(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    # the tests share the oscillator A and the sets X0, U
    def test_equality_compares_the_matrices_entrywise(self):
        B = np.array([[1.0], [0.5], [0.0]])
        # == compares the matrices entry by entry
        sys = cora.LinearSys(A, B)
        self.assertTrue(sys == cora.LinearSys(A.copy(), B.copy()))
        # a different input matrix is a different system
        self.assertTrue(sys != cora.LinearSys(A, 2 * B))
        self.assertTrue(sys.isequal(cora.LinearSys(A + 1e-3, B), tol=1e-1))
        self.assertFalse(sys.isequal(cora.LinearSys(A + 1e-3, B)))
        # CORA calls a system with permuted columns of A equal
        self.assertTrue(cora.LinearSys(A) != cora.LinearSys(A[:, [1, 0, 2]]))

    # the printout names the matrices that are present
    def test_repr_lists_the_matrices(self):
        text = repr(cora.LinearSys(A, np.array([[1.0], [0.5], [0.0]]), np.eye(3)))
        self.assertIn("System matrix", text)
        self.assertIn("Input matrix", text)
        self.assertIn("Output matrix", text)

    def test_generate_random_is_repeatable_and_stable(self):
        a = cora.LinearSys.generate_random(cora.Rng(3), 5, 2, 1)
        b = cora.LinearSys.generate_random(cora.Rng(3), 5, 2, 1)
        self.assertTrue(a == b)
        self.assertEqual((a.A.shape, a.B.shape, a.C.shape), ((5, 5), (5, 2), (1, 5)))
        self.assertLess(np.linalg.eigvals(a.A).real.max(), 0)

    def test_nonlinear_sys_has_the_same_flow(self):
        nl = cora.LinearSys(A).nonlinear_sys()
        self.assertEqual(nl.dim, 3)
        x = np.array([[1.0], [2.0], [3.0]])
        states = nl.simulate(initial_state=x, t_final=0.5, time_step=0.01)
        np.testing.assert_allclose(states[-1], expm(A * 0.5) @ x, rtol=1e-6)
        with self.assertRaises(cora.errors.CORAError):
            cora.LinearSys(A, np.eye(3)).nonlinear_sys()

    def test_from_start_and_decomposition_enclose_the_simulations(self):
        # simulations every half step: the even samples are time points, the odd ones inside a step
        sys = cora.LinearSys(A, np.eye(3))
        rng = cora.Rng(11)
        points = X0.rand_point(30, rng, "standard")
        inputs = U.rand_point(30, rng, "standard")
        x = sys.simulate(points, time_step=0.25, t_final=3.0, input=inputs)
        for alg, options in (("from_start", {}), ("decomposition", {"partition": [[0, 1], [2, 2]]})):
            R = sys.reach(X0, t_final=3.0, input_set=U, alg=alg, time_step=0.5, taylor_terms=8, **options)
            self.assertEqual((len(R.time_interval), len(R.time_point)), (6, 7))
            for k in range(13):
                sets = R.time_point[k // 2] if k % 2 == 0 else R.time_interval[k // 2]
                self.assertTrue(sets.contains(x[k]))
        with self.assertRaises(cora.errors.CORAError):
            sys.reach(X0, t_final=1.0, alg="decomposition", time_step=0.1)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
