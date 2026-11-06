"""test_nonlinear_sys - NonlinearSys, Expr and Zonotope.reduce in Python.

CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contDynamics/@contDynamics/reach.m
# CORA: contDynamics/@contDynamics/simulateRandom.m
# CORA: contDynamics/@nonlinearSys/nonlinearSys.m
# CORA: contDynamics/@nonlinearSys/simulate.m
# CORA: contSet/@zonotope/generateRandom.m
# CORA: contSet/@zonotope/reduce.m

import unittest

import numpy as np


try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch") from None

import cora


if not cora.has_torch:  # torch is installed, but this build has no torch backend
    raise unittest.SkipTest("needs a build with torch")

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def van_der_pol():
    return cora.NonlinearSys(lambda x: [x[1], (1 - x[0] ** 2) * x[1] - x[0]], 2)


def start():
    return cora.Zonotope(cora.Tensor([1.4, 2.3]), 0.05 * cora.eye(2))


# ============================================  MAIN  ============================================ #

class Expressions(unittest.TestCase):
    def test_python_operators_build_expressions(self):
        x = cora.Expr.var(0)
        e = 2 * x - 1 / (x + 3) + x ** 2 + cora.sin(x) * cora.cos(x) - cora.exp(-x)
        v = 0.7
        expected = 2 * v - 1 / (v + 3) + v ** 2 + np.sin(v) * np.cos(v) - np.exp(-v)
        self.assertAlmostEqual(e.eval([v]), expected, places=12)

    def test_the_derivative_is_symbolic(self):
        x = cora.Expr.var(0)
        self.assertAlmostEqual((x ** 3).diff(0).eval([2.0]), 12.0, places=12)
        self.assertAlmostEqual(cora.sin(x).diff(0).eval([0.0]), 1.0, places=12)

    def test_dynamics_may_return_numbers(self):
        sys = cora.NonlinearSys(lambda x: [x[1], 0.0], 2)
        x = sys.simulate(cora.Tensor([[0.0], [1.0]]), time_step=0.1, t_final=1.0)
        self.assertAlmostEqual(float(x[-1][0, 0]), 1.0, places=9)

    def test_wrong_dynamics_are_described(self):
        with self.assertRaises(ValueError) as caught:
            cora.NonlinearSys(lambda x: [x[0]], 2)
        self.assertIn("1 components for a system of dimension 2", str(caught.exception))


class Simulate(unittest.TestCase):
    # the harmonic oscillator
    def test_follows_the_harmonic_oscillator(self):
        sys = cora.NonlinearSys(lambda x: [x[1], -x[0]], 2)
        x = sys.simulate(cora.Tensor([[1.0], [0.0]]), time_step=0.1, t_final=2.0)
        self.assertEqual(tuple(x.shape), (21, 2, 1))
        np.testing.assert_allclose(np.asarray(x[-1]).ravel(), [np.cos(2.0), -np.sin(2.0)], atol=1e-9)

    def test_numpy_agrees_with_torch(self):
        x0 = np.array([[1.4], [2.3]])
        on_eigen = van_der_pol().simulate(x0, time_step=0.05, t_final=1.0)
        on_torch = van_der_pol().simulate(torch.tensor(x0), time_step=0.05, t_final=1.0)
        self.assertIsInstance(on_eigen, np.ndarray)
        np.testing.assert_allclose(on_eigen, on_torch.numpy(), atol=1e-12)

    def test_random_points_are_repeatable(self):
        a = van_der_pol().simulate_random(start(), points=4, time_step=0.05, t_final=0.5, rng=cora.Rng(2))
        b = van_der_pol().simulate_random(start(), points=4, time_step=0.05, t_final=0.5, rng=cora.Rng(2))
        self.assertEqual(tuple(a.shape), (11, 2, 4))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class ReachSet(unittest.TestCase):
    def test_simulations_stay_in_the_enclosures(self):
        R = van_der_pol().reach(start(), time_step=0.005, t_final=1.0)
        # random simulations stay in the reachable sets
        x = np.asarray(van_der_pol().simulate_random(start(), points=20, time_step=0.005, t_final=1.0,
                                                     rng=cora.Rng(3)))
        self.assertEqual(len(R.time_point), len(R.time_interval) + 1)
        for k, Z in enumerate(R.time_point):
            box = Z.interval()
            self.assertTrue(np.all(x[k] >= np.asarray(box.inf)[:, None] - 1e-9))
            self.assertTrue(np.all(x[k] <= np.asarray(box.sup)[:, None] + 1e-9))

    def test_the_zonotope_order_limits_the_generators(self):
        R = van_der_pol().reach(start(), time_step=0.01, t_final=0.5, zonotope_order=3)
        self.assertLessEqual(max(np.asarray(Z.generators).shape[1] for Z in R.time_point), 6)

    def test_a_step_that_is_too_large_is_described(self):
        with self.assertRaises(RuntimeError) as caught:
            van_der_pol().reach(start(), time_step=5.0, t_final=10.0)
        self.assertIn("too large", str(caught.exception))

    def test_reduce_encloses_the_zonotope(self):
        Z = cora.Zonotope.generate_random(cora.Rng(1), 3, nr_generators=20)
        R = Z.reduce(2)
        self.assertEqual(np.asarray(R.generators).shape, (3, 6))
        for k in range(20):
            d = torch.randn(3, dtype=torch.float64, generator=torch.Generator().manual_seed(k))
            # the support functions of the reachable set
            self.assertGreaterEqual(float(R.support_function(d)), float(Z.support_function(d)) - 1e-9)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
