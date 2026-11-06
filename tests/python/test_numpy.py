"""test_numpy - the Python package on numpy and Eigen alone; every test runs without torch.

Authors:       Claude Sonnet 5.5
Written:       30-September-2026
Last update:   ---
Reviewed:      ---
"""
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

class NumpyOnly(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_a_tensor_on_eigen_is_numpy(self):
        self.assertEqual(cora.current_backend(), "eigen")
        self.assertIsInstance(cora.Tensor([1.0, 2.0]), np.ndarray)

    def test_a_torch_backend_is_refused_or_works(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(cora.errors.Install, "has no torch"):
                with cora.backend("torch"):
                    pass

    def test_zonotope_and_interval_take_and_give_numpy(self):
        Z = cora.Zonotope(np.array([1.0, 2.0]), np.array([[0.5, 0.0], [0.0, 0.25]]))
        self.assertIsInstance(Z.center, np.ndarray)
        np.testing.assert_allclose(Z.interval().inf, [0.5, 1.75])
        np.testing.assert_allclose(Z.support_function(np.array([1.0, 1.0])), 3.75)
        M = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose((M @ Z).center, [2.0, -1.0])
        np.testing.assert_allclose((Z + np.array([1.0, 1.0])).center, [2.0, 3.0])
        I = cora.Interval(np.array([-1.0, 0.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(I.rad, [1.0, 1.0])
        self.assertTrue(I.contains(np.array([0.0, 1.0])))

    def test_linear_reach_and_simulate(self):
        sys = cora.LinearSys(np.array([[-0.5, 1.0], [-1.0, -0.5]]))
        X0 = cora.Zonotope(np.array([1.0, 0.0]), 0.1 * np.eye(2))
        R = sys.reach(X0, time_step=0.1, t_final=1.0)
        x = sys.simulate(np.array([[1.0], [0.0]]), time_step=0.1, t_final=1.0)
        # the reachable sets contain the simulations
        self.assertEqual(x.shape, (11, 2, 1))
        for k, Z in enumerate(R.time_point):
            box = Z.interval()
            self.assertTrue(np.all(box.inf <= x[k, :, 0] + 1e-9))
            self.assertTrue(np.all(x[k, :, 0] <= box.sup + 1e-9))

    def test_nonlinear_reach(self):
        sys = cora.NonlinearSys(lambda x: [x[1], -x[0]], 2)
        X0 = cora.Zonotope(np.array([1.0, 0.0]), 0.05 * np.eye(2))
        self.assertEqual(len(sys.reach(X0, time_step=0.05, t_final=0.5).time_interval), 10)
        self.assertEqual(sys.simulate(np.array([[1.0], [0.0]]), time_step=0.05, t_final=0.5).shape, (11, 2,
                                                                                                     1))

    def test_neural_network_and_specification(self):
        W, b = np.array([[1.0, -1.0], [0.5, 2.0]]), np.array([0.1, -0.2])
        net = cora.NeuralNetwork([(W, b)])
        x = np.array([0.3, 0.4])
        np.testing.assert_allclose(net.evaluate(x), W @ x + b)
        Y = net.evaluate(cora.Zonotope(x, 0.1 * np.eye(2)))
        spec = cora.Specification.safe_set(np.array([1.0, 0.0]), 10.0)
        self.assertTrue(spec.check(Y))

    def test_plot_takes_numpy_points(self):
        # plotting arrays
        import matplotlib
        matplotlib.use("Agg")
        cora.plot(np.array([[0.0, 1.0], [0.0, 1.0]]), dims=[0, 1])


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
