"""test_plot - cora.plot draws what the C++ plotting core decided, with matplotlib"""
import unittest

import matplotlib

matplotlib.use("Agg")
import cora
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch


def face(patch):
    """The RGB of the fill of a patch."""
    return tuple(float(v) for v in patch.get_facecolor()[:3])


def paths(ax):
    """The patches CORA drew for sets: one compound path per layer (specifications are polygons)."""
    return [p for p in ax.patches if isinstance(p, PathPatch)]


def regions(ax):
    """The patches of specifications."""
    return [p for p in ax.patches if not isinstance(p, PathPatch)]


class Plot(unittest.TestCase):
    def setUp(self):
        cora.use_cora_colors("CORA:default")
        self.fig, self.ax = plt.subplots()
        self.sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))
        self.X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

    def tearDown(self):
        cora.use_cora_colors("CORA:default")
        plt.close(self.fig)

    def test_a_simulation_is_one_line_per_trajectory_and_nothing_else(self):
        x = self.sys.simulate(self.X0.rand_point(4, cora.Rng(1)), time_step=0.1, t_final=2.0)
        cora.plot(x, ax=self.ax)
        self.assertEqual(len(self.ax.lines), 4)
        self.assertEqual(self.ax.lines[0].get_marker(), "None")

    def test_sets_take_the_colors_of_cora_in_turn(self):
        for _ in range(3):
            cora.plot(self.X0, ax=self.ax)
        colors = [face(c) for c in paths(self.ax)]
        self.assertEqual(colors, [cora.cora_color(f"CORA:{n}") for n in ("blue", "red", "yellow")])

    def test_a_set_is_filled_and_an_axes_has_its_own_order(self):
        cora.plot(self.X0, ax=self.ax)
        other = self.fig.add_subplot(2, 1, 2)
        cora.plot(self.X0, ax=other)
        self.assertEqual(face(paths(other)[0]), cora.cora_color("CORA:blue"))
        self.assertGreater(paths(self.ax)[0].get_facecolor()[3], 0.0)

    def test_a_filled_set_is_translucent_and_its_outline_is_not(self):
        cora.plot(self.X0, ax=self.ax)
        cora.plot(self.X0, ax=self.ax, face_alpha=0.5)
        first, second = paths(self.ax)
        self.assertAlmostEqual(float(first.get_facecolor()[3]), 0.2)
        self.assertAlmostEqual(float(second.get_facecolor()[3]), 0.5)
        self.assertEqual(float(first.get_edgecolor()[3]), 1.0)

    def test_corners_are_sharp(self):
        cora.plot(self.X0, ax=self.ax)
        self.assertEqual(paths(self.ax)[0].get_joinstyle(), "miter")

    def test_the_dynamics_scheme_gives_initial_sets_and_blue_reachable_sets(self):
        cora.use_cora_colors("CORA:contDynamics")
        cora.plot(self.X0, ax=self.ax)
        cora.plot(self.sys.reach(self.X0, time_step=0.1, t_final=1.0), ax=self.ax)
        initial, reach = paths(self.ax)
        self.assertEqual(face(initial), cora.cora_color("CORA:initialSet"))
        self.assertEqual(tuple(initial.get_edgecolor()[:3]), cora.cora_color("CORA:simulations"))
        self.assertEqual(face(reach), cora.cora_color("CORA:reachSet"))
        self.assertGreater(reach.get_zorder(), 0)
        self.assertLess(reach.get_zorder(), initial.get_zorder())

    def test_a_set_is_open_without_a_fill(self):
        cora.plot(self.X0, ax=self.ax, facecolor="none")
        cora.plot(self.X0, ax=self.ax, filled=False)
        for patch in paths(self.ax):
            self.assertFalse(patch.get_fill())

    def test_a_color_is_a_cora_identifier_an_rgb_triple_or_a_matplotlib_color(self):
        cora.plot(self.X0, ax=self.ax, color="CORA:green")
        cora.plot(self.X0, ax=self.ax, color=(0.1, 0.2, 0.3))
        cora.plot(self.X0, ax=self.ax, color="red")
        self.assertEqual(face(paths(self.ax)[0]), cora.cora_color("CORA:green"))
        self.assertEqual(face(paths(self.ax)[1]), (0.1, 0.2, 0.3))
        self.assertEqual(face(paths(self.ax)[2]), (1.0, 0.0, 0.0))

    def test_other_keywords_go_to_matplotlib(self):
        cora.plot(self.X0, ax=self.ax, alpha=0.5)
        self.assertEqual(paths(self.ax)[0].get_alpha(), 0.5)
        with self.assertRaises(AttributeError):
            cora.plot(self.X0, ax=self.ax, no_such_property=1)

    def test_a_label_reaches_the_legend(self):
        cora.plot(self.X0, ax=self.ax, label="Initial set")
        self.assertEqual(self.ax.get_legend_handles_labels()[1], ["Initial set"])

    def test_a_reachable_set_is_one_region(self):
        cora.plot(self.sys.reach(self.X0, time_step=0.1, t_final=1.0), ax=self.ax)
        # the sets are united in the core: one outline, drawn as one path
        self.assertEqual(len(paths(self.ax)), 1)

    def test_time_points_and_step_are_options_of_the_core(self):
        R = self.sys.reach(self.X0, time_step=0.1, t_final=1.0)
        cora.plot(R, ax=self.ax, time_points=True, step=2)
        reach, points = paths(self.ax)
        self.assertFalse(points.get_fill())
        # every second set: fewer corners than with every set
        every = len(points.get_path().vertices)
        cora.plot(R, ax=self.ax, time_points=True, step=1)
        self.assertGreater(len(paths(self.ax)[3].get_path().vertices), every)

    def test_a_specification_is_a_region_up_to_the_limits(self):
        cora.plot(self.X0, ax=self.ax)
        limits = (self.ax.get_xlim(), self.ax.get_ylim())
        cora.plot(cora.Specification.unsafe_set(cora.Tensor([1.0, 0.0]), 1.0), ax=self.ax)
        self.assertEqual(len(regions(self.ax)), 1)
        self.assertEqual((self.ax.get_xlim(), self.ax.get_ylim()), limits)

    def test_something_unknown_is_described(self):
        with self.assertRaises(TypeError) as caught:
            cora.plot("not a set", ax=self.ax)
        self.assertIn("does not know how to draw", str(caught.exception))

    def test_a_wrong_option_of_the_core_is_described(self):
        with self.assertRaises(ValueError) as caught:
            cora.plot(self.X0, ax=self.ax, step=0, time_points=True) if False else \
                cora.plot(self.sys.reach(self.X0, time_step=0.1, t_final=0.5), ax=self.ax, step=0)
        self.assertIn("step must be at least 1", str(caught.exception))


class Dimensions(unittest.TestCase):
    """One, two and three dimensions, over time, as methods, and the options of CORA's plot."""

    def setUp(self):
        cora.use_cora_colors("CORA:default")
        self.sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))
        self.X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))
        self.Z3 = cora.Zonotope(cora.Tensor([0.0, 0.0, 0.0]),
                                cora.Tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]))

    def tearDown(self):
        cora.use_cora_colors("CORA:default")
        plt.close("all")

    def test_one_dimension_is_a_bar(self):
        _, ax = plt.subplots()
        cora.plot(self.X0, dims=(1,), ax=ax)
        (patch,) = paths(ax)
        self.assertEqual(np.asarray(patch.get_path().vertices)[:2].tolist(), [[-0.1, 0.0], [0.1, 0.0]])

    def test_three_dimensions_are_the_faces_of_the_polytope(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        cora.plot(self.Z3, dims=(0, 1, 2), ax=ax, label="Z")
        self.assertEqual(len(ax.collections), 1)
        (layer,) = ax._cora_figure.layers()
        self.assertEqual((layer["kind"], len(layer["faces"])), ("surface", 6))
        self.assertEqual(ax.get_zlabel(), "x[2]")

    def test_plotting_three_dimensions_without_axes_makes_them(self):
        plt.close("all")
        cora.plot(self.Z3, dims=(0, 1, 2))
        self.assertEqual(plt.gca().name, "3d")

    def test_a_plot_over_time_has_time_on_the_x_axis(self):
        _, ax = plt.subplots()
        R = self.sys.reach(self.X0, time_step=0.1, t_final=1.0)
        cora.plot_over_time(R, 0, ax=ax)
        self.assertEqual(ax.get_xlabel(), "t")
        x = paths(ax)[0].get_path().vertices[:, 0]
        self.assertAlmostEqual(float(x.min()), 0.0)
        self.assertAlmostEqual(float(x.max()), 1.0)
        R.plot_over_time(1, ax=ax, set="tp")  # the time points as lines: a method
        self.assertEqual(len(paths(ax)), 2)

    def test_simulations_over_time_need_their_time_step(self):
        _, ax = plt.subplots()
        x = self.sys.simulate(self.X0.rand_point(3, cora.Rng(1)), time_step=0.1, t_final=1.0)
        cora.plot_over_time(x, 0, ax=ax, time_step=0.1)
        self.assertEqual(len(ax.lines), 3)
        self.assertAlmostEqual(float(ax.lines[0].get_xdata()[-1]), 1.0, places=6)

    def test_the_plot_functions_are_methods(self):
        _, ax = plt.subplots()
        self.X0.plot(ax=ax)
        R = self.sys.reach(self.X0, time_step=0.1, t_final=1.0)
        R.plot(ax=ax)
        cora.Specification.unsafe_set(cora.Tensor([1.0, 0.0]), 1.0).plot(ax=ax)
        self.X0.plot_rand_point(ax=ax, num_points=7)
        self.assertEqual(len(paths(ax)), 2)
        self.assertEqual(len(regions(ax)), 1)
        self.assertEqual(len(ax.lines[-1].get_xdata()), 7)

    def test_edge_and_face_colors_and_the_set_of_a_reach_plot(self):
        _, ax = plt.subplots()
        cora.plot(self.X0, ax=ax, face_color="CORA:red", edge_color="CORA:green")
        patch = paths(ax)[0]
        self.assertEqual(face(patch), cora.cora_color("CORA:red"))
        self.assertEqual(tuple(float(v) for v in patch.get_edgecolor()[:3]),
                         cora.cora_color("CORA:green"))
        R = self.sys.reach(self.X0, time_step=0.1, t_final=1.0)
        cora.plot(R, ax=ax, set="tp")
        with self.assertRaises(ValueError):
            cora.plot(R, ax=ax, set="xx")

    def test_a_network_input_and_output_set_are_sets(self):
        net = cora.NeuralNetwork([(cora.Tensor([[1.0, -1.0], [0.5, 2.0]]), cora.Tensor([0.0, 0.0]))])
        _, ax = plt.subplots()
        cora.plot(self.X0, ax=ax)
        cora.plot(net.evaluate(self.X0), ax=ax)
        net_points = net.evaluate(self.X0.rand_point(5, cora.Rng(2)))
        cora.plot(net_points, ax=ax)
        self.assertEqual(len(paths(ax)), 2)
        self.assertEqual(len(ax.lines), 1)

    def test_a_python_set_plots_in_three_dimensions_through_vertices_3d(self):
        class Cube(cora.ContSet):
            def _dim(self): return 3
            def _center(self): return cora.Tensor([0.0, 0.0, 0.0])
            def _interval(self): return cora.Interval(cora.Tensor([-1.0] * 3), cora.Tensor([1.0] * 3))
            def _project(self, dims):
                return Cube() if list(dims) == [0, 1, 2] else self._interval().project(dims)
            def _vertices_3d(self):
                return [np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)]
        cube = Cube()
        self.assertEqual(np.asarray(cube.vertices_3d()[0]).shape, (8, 3))
        ax = plt.figure().add_subplot(projection="3d")
        cora.plot(cube, dims=(0, 1, 2), ax=ax)
        self.assertEqual(len(ax.collections), 1)

    def test_vertices_3d_of_the_sets(self):
        box = cora.Interval(cora.Tensor([0.0, 0.0, 0.0]), cora.Tensor([1.0, 2.0, 3.0]))
        (corners,) = box.vertices_3d()
        self.assertEqual(np.asarray(corners).shape, (8, 3))
        (corners,) = self.Z3.vertices_3d()
        self.assertEqual(np.asarray(corners).shape, (8, 3))
        with self.assertRaises(cora.errors.NotSupported):
            self.X0.vertices_3d()


class Colors(unittest.TestCase):
    def test_cora_colors_come_from_the_core(self):
        self.assertAlmostEqual(cora.cora_color("CORA:blue")[1], 0.4470)
        self.assertEqual(cora.cora_color(2), cora.cora_color("CORA:red"))
        self.assertAlmostEqual(cora.cora_color("CORA:reachSet", num_colors=3, cidx=3)[0], 0.2706)

    def test_unknown_colors_and_schemes_are_described(self):
        with self.assertRaises(ValueError):
            cora.cora_color("CORA:nope")
        with self.assertRaises(ValueError) as caught:
            cora.use_cora_colors("CORA:rainbow")
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
