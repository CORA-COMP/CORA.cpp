"""test_plot - cora.plot draws what the C++ plotting core decided, with matplotlib"""
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cora


def face(collection):
    """The RGB of the fill of a collection with one fill color."""
    return tuple(float(v) for v in collection.get_facecolor()[0][:3])


class Plot(unittest.TestCase):
    def setUp(self):
        cora.useCORAcolors("CORA:default")
        self.fig, self.ax = plt.subplots()
        self.sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))
        self.X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

    def tearDown(self):
        cora.useCORAcolors("CORA:default")
        plt.close(self.fig)

    def test_a_simulation_is_one_line_per_trajectory_and_nothing_else(self):
        x = self.sys.simulate(self.X0.randPoint(4, cora.Rng(1)), 0.1, 2.0)
        cora.plot(x, ax=self.ax)
        self.assertEqual(len(self.ax.lines), 4)
        self.assertEqual(self.ax.lines[0].get_marker(), "None")

    def test_sets_take_the_colors_of_cora_in_turn(self):
        for _ in range(3):
            cora.plot(self.X0, ax=self.ax)
        colors = [face(c) for c in self.ax.collections]
        self.assertEqual(colors, [cora.CORAcolor(f"CORA:{n}") for n in ("blue", "red", "yellow")])

    def test_a_set_is_filled_and_an_axes_has_its_own_order(self):
        cora.plot(self.X0, ax=self.ax)
        other = self.fig.add_subplot(2, 1, 2)
        cora.plot(self.X0, ax=other)
        self.assertEqual(face(other.collections[0]), cora.CORAcolor("CORA:blue"))
        self.assertGreater(self.ax.collections[0].get_facecolor()[0][3], 0.0)

    def test_a_filled_set_is_translucent_and_its_outline_is_not(self):
        cora.plot(self.X0, ax=self.ax)
        cora.plot(self.X0, ax=self.ax, faceAlpha=0.5)
        first, second = self.ax.collections
        self.assertAlmostEqual(float(first.get_facecolor()[0][3]), 0.2)
        self.assertAlmostEqual(float(second.get_facecolor()[0][3]), 0.5)
        self.assertEqual(float(first.get_edgecolor()[0][3]), 1.0)

    def test_corners_are_sharp(self):
        cora.plot(self.X0, ax=self.ax)
        self.assertEqual(self.ax.collections[0].get_joinstyle(), "miter")

    def test_the_dynamics_scheme_gives_initial_sets_and_blue_reachable_sets(self):
        cora.useCORAcolors("CORA:contDynamics")
        cora.plot(self.X0, ax=self.ax)
        cora.plot(self.sys.reach(self.X0, 0.1, 1.0), ax=self.ax)
        initial, reach = self.ax.collections
        self.assertEqual(face(initial), cora.CORAcolor("CORA:initialSet"))
        self.assertEqual(tuple(initial.get_edgecolor()[0][:3]), cora.CORAcolor("CORA:simulations"))
        self.assertEqual(face(reach), cora.CORAcolor("CORA:reachSet"))
        self.assertGreater(reach.get_zorder(), 0)
        self.assertLess(reach.get_zorder(), initial.get_zorder())

    def test_a_set_is_open_without_a_fill(self):
        cora.plot(self.X0, ax=self.ax, facecolor="none")
        cora.plot(self.X0, ax=self.ax, filled=False)
        for collection in self.ax.collections:
            self.assertEqual(len(collection.get_facecolor()), 0)

    def test_a_color_is_a_cora_identifier_an_rgb_triple_or_a_matplotlib_color(self):
        cora.plot(self.X0, ax=self.ax, color="CORA:green")
        cora.plot(self.X0, ax=self.ax, color=(0.1, 0.2, 0.3))
        cora.plot(self.X0, ax=self.ax, color="red")
        self.assertEqual(face(self.ax.collections[0]), cora.CORAcolor("CORA:green"))
        self.assertEqual(face(self.ax.collections[1]), (0.1, 0.2, 0.3))
        self.assertEqual(face(self.ax.collections[2]), (1.0, 0.0, 0.0))

    def test_other_keywords_go_to_matplotlib(self):
        cora.plot(self.X0, ax=self.ax, alpha=0.5)
        self.assertEqual(self.ax.collections[0].get_alpha(), 0.5)
        with self.assertRaises(AttributeError):
            cora.plot(self.X0, ax=self.ax, no_such_property=1)

    def test_a_label_reaches_the_legend(self):
        cora.plot(self.X0, ax=self.ax, label="Initial set")
        self.assertEqual(self.ax.get_legend_handles_labels()[1], ["Initial set"])

    def test_a_reachable_set_is_one_region(self):
        cora.plot(self.sys.reach(self.X0, 0.1, 1.0), ax=self.ax)
        self.assertEqual(len(self.ax.collections), 1)
        self.assertEqual(len(self.ax.collections[0].get_paths()), 10)

    def test_time_points_and_step_are_options_of_the_core(self):
        R = self.sys.reach(self.X0, 0.1, 1.0)
        cora.plot(R, ax=self.ax, timePoints=True, step=2)
        reach, points = self.ax.collections
        self.assertEqual(len(reach.get_paths()), 5)
        self.assertEqual(len(points.get_paths()), 6)
        self.assertEqual(len(points.get_facecolor()), 0)

    def test_a_specification_is_a_region_up_to_the_limits(self):
        cora.plot(self.X0, ax=self.ax)
        limits = (self.ax.get_xlim(), self.ax.get_ylim())
        cora.plot(cora.Specification.unsafeSet(cora.Tensor([1.0, 0.0]), 1.0), ax=self.ax)
        self.assertEqual(len(self.ax.patches), 1)
        self.assertEqual((self.ax.get_xlim(), self.ax.get_ylim()), limits)

    def test_something_unknown_is_described(self):
        with self.assertRaises(TypeError) as caught:
            cora.plot("not a set", ax=self.ax)
        self.assertIn("does not know how to draw", str(caught.exception))

    def test_a_wrong_option_of_the_core_is_described(self):
        with self.assertRaises(ValueError) as caught:
            cora.plot(self.X0, ax=self.ax, step=0, timePoints=True) if False else \
                cora.plot(self.sys.reach(self.X0, 0.1, 0.5), ax=self.ax, step=0)
        self.assertIn("step must be at least 1", str(caught.exception))


class Colors(unittest.TestCase):
    def test_cora_colors_come_from_the_core(self):
        self.assertAlmostEqual(cora.CORAcolor("CORA:blue")[1], 0.4470)
        self.assertEqual(cora.CORAcolor(2), cora.CORAcolor("CORA:red"))
        self.assertAlmostEqual(cora.CORAcolor("CORA:reachSet", numColors=3, cidx=3)[0], 0.2706)

    def test_unknown_colors_and_schemes_are_described(self):
        with self.assertRaises(ValueError):
            cora.CORAcolor("CORA:nope")
        with self.assertRaises(ValueError) as caught:
            cora.useCORAcolors("CORA:rainbow")
        self.assertIn("CORA:contDynamics", str(caught.exception))


class Display(unittest.TestCase):
    def test_a_set_prints_as_text(self):
        Z = cora.Zonotope(cora.Tensor([-1.0, 0.0]), 0.01 * cora.eye(2))
        self.assertIn("zonotope", str(Z))
        self.assertIn("- generators (2):", repr(Z))
        I = cora.Interval(cora.Tensor([0.0, 1.0]), cora.Tensor([2.0, 3.0]))
        self.assertIn("[0, 2]", str(I))

    def test_vertices_are_arrays_per_set(self):
        box = cora.Interval(cora.Tensor([0.0, 0.0]), cora.Tensor([2.0, 1.0]))
        (corners,) = box.vertices()
        self.assertEqual(np.asarray(corners).shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
