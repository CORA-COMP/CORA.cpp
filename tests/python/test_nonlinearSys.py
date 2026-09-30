"""test_nonlinearSys - NonlinearSys, Expr and Zonotope.reduce in Python."""
import unittest

import numpy as np
try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch")

import cora


def vanDerPol():
    return cora.NonlinearSys(lambda x: [x[1], (1 - x[0] ** 2) * x[1] - x[0]], 2)


def start():
    return cora.Zonotope(cora.Tensor([1.4, 2.3]), 0.05 * cora.eye(2))


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
        x = sys.simulate(cora.Tensor([[0.0], [1.0]]), 0.1, 1.0)
        self.assertAlmostEqual(float(x[-1][0, 0]), 1.0, places=9)

    def test_wrong_dynamics_are_described(self):
        with self.assertRaises(ValueError) as caught:
            cora.NonlinearSys(lambda x: [x[0]], 2)
        self.assertIn("1 components for a system of dimension 2", str(caught.exception))


class Simulate(unittest.TestCase):
    def test_follows_the_harmonic_oscillator(self):
        sys = cora.NonlinearSys(lambda x: [x[1], -x[0]], 2)
        x = sys.simulate(cora.Tensor([[1.0], [0.0]]), 0.1, 2.0)
        self.assertEqual(tuple(x.shape), (21, 2, 1))
        np.testing.assert_allclose(np.asarray(x[-1]).ravel(), [np.cos(2.0), -np.sin(2.0)], atol=1e-9)

    def test_numpy_agrees_with_torch(self):
        x0 = np.array([[1.4], [2.3]])
        on_eigen = vanDerPol().simulate(x0, 0.05, 1.0)
        on_torch = vanDerPol().simulate(torch.tensor(x0), 0.05, 1.0)
        self.assertIsInstance(on_eigen, np.ndarray)
        np.testing.assert_allclose(on_eigen, on_torch.numpy(), atol=1e-12)

    def test_random_points_are_repeatable(self):
        a = vanDerPol().simulateRandom(start(), 4, 0.05, 0.5, cora.Rng(2))
        b = vanDerPol().simulateRandom(start(), 4, 0.05, 0.5, cora.Rng(2))
        self.assertEqual(tuple(a.shape), (11, 2, 4))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class Reach(unittest.TestCase):
    def test_simulations_stay_in_the_enclosures(self):
        R = vanDerPol().reach(start(), 0.005, 1.0)
        x = np.asarray(vanDerPol().simulateRandom(start(), 20, 0.005, 1.0, cora.Rng(3)))
        self.assertEqual(len(R.timePoint), len(R.timeInt) + 1)
        for k, Z in enumerate(R.timePoint):
            box = Z.interval()
            self.assertTrue(np.all(x[k] >= np.asarray(box.inf)[:, None] - 1e-9))
            self.assertTrue(np.all(x[k] <= np.asarray(box.sup)[:, None] + 1e-9))

    def test_the_zonotope_order_limits_the_generators(self):
        R = vanDerPol().reach(start(), 0.01, 0.5, zonotopeOrder=3)
        self.assertLessEqual(max(np.asarray(Z.G).shape[1] for Z in R.timePoint), 6)

    def test_a_step_that_is_too_large_is_described(self):
        with self.assertRaises(RuntimeError) as caught:
            vanDerPol().reach(start(), 5.0, 10.0)
        self.assertIn("too large", str(caught.exception))

    def test_reduce_encloses_the_zonotope(self):
        Z = cora.Zonotope.generateRandom(3, 20, cora.Rng(1))
        R = Z.reduce(2)
        self.assertEqual(np.asarray(R.G).shape, (3, 6))
        for k in range(20):
            d = torch.randn(3, dtype=torch.float64, generator=torch.Generator().manual_seed(k))
            self.assertGreaterEqual(float(R.supportFunc(d)), float(Z.supportFunc(d)) - 1e-9)


if __name__ == "__main__":
    unittest.main()
