"""test_plot - cora.plot draws what it is given, and simulations carry no start markers"""
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cora


class Plot(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))
        self.X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

    def tearDown(self):
        plt.close(self.fig)

    def test_a_simulation_is_one_line_per_trajectory_and_nothing_else(self):
        x = self.sys.simulate(self.X0.randPoint(4, cora.Rng(1)), 0.1, 2.0)
        cora.plot(x, ax=self.ax)
        self.assertEqual(len(self.ax.lines), 4)

    def test_a_reachable_set_is_drawn_as_patches(self):
        cora.plot(self.sys.reach(self.X0, 0.1, 1.0), ax=self.ax)
        self.assertEqual(len(self.ax.patches), 10)

    def test_something_unknown_is_described(self):
        with self.assertRaises(TypeError) as caught:
            cora.plot("not a set", ax=self.ax)
        self.assertIn("does not know how to draw", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
