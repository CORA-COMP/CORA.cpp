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

    def test_a_set_is_an_initial_set_by_default(self):
        cora.plot(self.X0, ax=self.ax)
        patch = self.ax.patches[0]
        self.assertEqual(patch.get_facecolor()[:3], cora.CORAcolor("CORA:initialSet"))
        self.assertEqual(patch.get_edgecolor()[:3], cora.CORAcolor("CORA:simulations"))

    def test_a_facecolor_replaces_the_default(self):
        cora.plot(self.X0, ax=self.ax, facecolor="none")
        self.assertEqual(self.ax.patches[0].get_facecolor()[3], 0.0)

    def test_there_is_no_special_function_for_the_initial_set(self):
        self.assertFalse(hasattr(cora, "plot_initial_set"))

    def test_a_reachable_set_is_drawn_as_patches(self):
        cora.plot(self.sys.reach(self.X0, 0.1, 1.0), ax=self.ax)
        self.assertEqual(len(self.ax.patches), 10)

    def test_something_unknown_is_described(self):
        with self.assertRaises(TypeError) as caught:
            cora.plot("not a set", ax=self.ax)
        self.assertIn("does not know how to draw", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
