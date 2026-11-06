"""test_linear_sys_steps - the steps of linear reach in Python: canonical form, the solutions of a step,

one_step, init_reach with post and the adaptive error of the input solution.

CORA authors: Mark Wetzlinger, Matthias Althoff
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@linearSys/canonicalForm.m
# CORA: contDynamics/@linearSys/isCanonicalForm.m
# CORA: contDynamics/@linearSys/homogeneousSolution.m
# CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
# CORA: contDynamics/@linearSys/particularSolution_constant.m
# CORA: contDynamics/@linearSys/affineSolution.m
# CORA: contDynamics/@linearSys/oneStep.m
# CORA: contDynamics/@linearSys/initReach.m
# CORA: contDynamics/@linearSys/post.m
# CORA: contDynamics/@linearSys/errorSolution_adaptive.m
# CORA: contDynamics/@linearSys/trajectoryError.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

A = np.diag([-1.0, -2.0])
X = cora.Zonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 0.5]]))
DT, TERMS = 0.1, 10


def expm(M):
    """The matrix exponential of a diagonalizable M."""
    w, V = np.linalg.eig(M)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real


def flow(t, x, u=0.0):
    """The state at t of x' = A x + u from x."""
    return expm(A * t) @ x + np.linalg.inv(A) @ (expm(A * t) - np.eye(2)) @ (np.zeros(2) + u)


# ============================================  MAIN  ============================================ #

class Steps(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.sys = cora.LinearSys(A)

    # the steps below are compared with the closed form of the diagonal system x' = A x + u
    def test_canonical_form_moves_b_into_the_sets(self):
        B = np.array([[1.0, 2.0], [0.0, 3.0]])
        U = cora.Zonotope(np.array([1.0, -1.0]), np.array([[0.5, 0.0], [0.0, 0.25]]))
        cf = cora.LinearSys(A, B).canonical_form(U, np.array([[0.5], [0.5]]))
        np.testing.assert_allclose(cf.U.center, 0.0)
        np.testing.assert_allclose(cf.U.generators, B @ U.generators)
        np.testing.assert_allclose(cf.u, B @ np.array([[1.5], [-0.5]]))
        self.assertTrue(cf.sys.is_canonical_form())
        self.assertFalse(cora.LinearSys(A, B).is_canonical_form())
        self.assertFalse(self.sys.is_canonical_form())
        with self.assertRaises(cora.errors.CORAError):
            self.sys.canonical_form(U)

    def test_homogeneous_solution_is_the_flow_of_the_set(self):
        H = self.sys.homogeneous_solution(X, DT, TERMS)
        np.testing.assert_allclose(H.time_point.center, expm(A * DT) @ X.center, atol=1e-14)
        for x in ([2.0, 2.5], [0.0, 1.5]):
            for t in (0.0, 0.05, DT):
                self.assertTrue(H.time_interval.contains(flow(t, np.array(x))))

    def test_particular_solutions(self):
        # the terms of the series stay apart for u(t), the closed form holds for a constant u
        U = cora.Zonotope(np.zeros(2), np.array([[1.0], [0.0]]))
        PU = self.sys.particular_solution_time_varying(U, DT, TERMS)
        self.assertAlmostEqual(PU.support_function(np.array([1.0, 0.0])), np.exp(DT) - 1, places=12)
        P = self.sys.particular_solution_constant(cora.Zonotope(np.ones(2), np.zeros((2, 0))), DT, TERMS)
        expected = [1 - np.exp(-DT), (1 - np.exp(-2 * DT)) / 2]
        np.testing.assert_allclose(P.time_point.center, expected, atol=1e-14)
        self.assertTrue(P.time_interval.contains(P.time_point.center))

    def test_affine_solution_and_one_step_agree(self):
        u = np.array([1.0, -1.0])
        S = self.sys.affine_solution(X, u, DT, TERMS)
        np.testing.assert_allclose(S.time_point.center, flow(DT, X.center, u), atol=1e-14)
        U = cora.Zonotope(np.zeros(2), np.array([[0.5], [0.25]]))
        step = self.sys.one_step(X, U, u, DT, TERMS)
        PU = self.sys.particular_solution_time_varying(U, DT, TERMS)
        np.testing.assert_allclose(step.time_point.center, S.time_point.center + PU.center, atol=1e-14)
        self.assertTrue(step.time_interval.contains(step.time_point.center))

    def test_init_reach_and_post_enclose_the_trajectories(self):
        # the end of the step k under a constant input of the sets, from a corner
        sys = cora.LinearSys(A, np.eye(2))
        U = cora.Zonotope(np.zeros(2), np.array([[0.1], [0.05]]))
        u = np.array([0.5, -0.5])
        for alg in ("standard", "wrapping_free"):
            step = sys.init_reach(X, U, u, alg=alg, time_step=DT, taylor_terms=TERMS)
            for k in range(1, 4):
                x = flow(k * DT, np.array([2.0, 2.5]), u + np.array([0.1, 0.05]))
                self.assertTrue(step.time_point.contains(x))
                self.assertTrue(step.time_interval.contains(x))
                step = sys.post(step.state, alg=alg, time_step=DT, taylor_terms=TERMS)
        with self.assertRaises(cora.errors.CORAError):
            sys.init_reach(X, alg="from_start", time_step=DT)

    # a forced order is the series; a smaller zeta_tabs keeps more terms
    def test_error_solution_is_cut_by_the_gain_or_the_order(self):
        V = cora.Zonotope(np.zeros(2), np.array([[1.0], [0.5]]))
        fixed = self.sys.error_solution_adaptive(V, time_step=DT, zeta_tabs=1e-3, taylor_terms=5)
        series = self.sys.particular_solution_time_varying(V, DT, 5)
        d = np.array([1.0, -2.0])
        self.assertAlmostEqual(fixed.support_function(d), series.support_function(d), places=12)
        # a smaller zeta_tabs keeps more terms, so more generators
        coarse = self.sys.error_solution_adaptive(V, time_step=DT, zeta_tabs=0.1)
        fine = self.sys.error_solution_adaptive(V, time_step=DT, zeta_tabs=1e-9)
        self.assertLess(coarse.generators.shape[1], fine.generators.shape[1])

    def test_trajectory_error_vanishes_at_the_expansion_point(self):
        B = np.array([[-1.0, 2.0], [-2.0, -1.0]])
        sys, t, n = cora.LinearSys(B), 0.3, np.array([1.0, -0.5])
        R0 = cora.Zonotope(np.array([1.0, 0.5]), 0.1 * np.eye(2))
        e = sys.trajectory_error([np.zeros(2), np.zeros(2)], np.zeros(2), R0, t, t, t, 12, n)
        self.assertLess(e.error.support_function(np.array([1.0, 1.0])), 1e-8)
        remainder = expm(B * t) - np.eye(2) - B * t
        np.testing.assert_allclose((e.c_tilde.inf + e.c_tilde.sup) / 2, remainder.T @ n, atol=1e-9)
        np.testing.assert_allclose((e.d_tilde.inf + e.d_tilde.sup) / 2, n @ remainder @ R0.center, atol=1e-9)
        with self.assertRaises(cora.errors.CORAError):
            sys.trajectory_error(np.zeros(2), np.zeros(2), R0, t, t, t, 0, n)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
