"""Plotting, as in CORA: coracpp.plot and its parts, the colors, and every example."""
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import coracpp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
F64 = dict(dtype=torch.float64)


def polygon_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex_counter_clockwise(v):
    edges = np.roll(v, -1, axis=0) - v
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    return bool(np.all(cross >= -1e-12))


def oscillator():
    A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **F64)
    return A, torch.tensor([1.0, 0.0], **F64), 0.1 * torch.eye(2, **F64)


class ZonotopeVertices(unittest.TestCase):
    def test_a_box(self):
        v = coracpp.zonotope_vertices([1.0, 2.0], [[1.0, 0.0], [0.0, 3.0]])
        self.assertEqual({tuple(p) for p in v.round(12)}, {(0.0, -1.0), (2.0, -1.0), (2.0, 5.0), (0.0, 5.0)})

    def test_random_zonotopes(self):
        rng = np.random.default_rng(0)
        for m in (1, 2, 3, 5, 8):
            c, G = rng.normal(size=2), rng.normal(size=(2, m))
            v = coracpp.zonotope_vertices(c, G)
            # The polygon of a zonotope has 2m vertices, is convex, and its area is
            # 4 * sum over pairs of |det(g_i, g_j)|.
            self.assertEqual(len(v), 2 * m)
            self.assertTrue(is_convex_counter_clockwise(v) or m == 1)
            area = 4 * sum(abs(np.linalg.det(G[:, [i, j]])) for i, j in itertools.combinations(range(m), 2))
            self.assertAlmostEqual(polygon_area(v), area, places=9)
            # Every vertex is a corner of the generator cube.
            corners = np.array([c + G @ np.array(b) for b in itertools.product([-1, 1], repeat=m)])
            for p in v:
                self.assertLess(np.min(np.linalg.norm(corners - p, axis=1)), 1e-9)

    def test_a_point_and_parallel_generators(self):
        self.assertEqual(coracpp.zonotope_vertices([1.0, 1.0], np.zeros((2, 2))).shape, (1, 2))
        v = coracpp.zonotope_vertices([0.0, 0.0], [[1.0, 2.0], [0.0, 0.0]])  # a segment
        self.assertAlmostEqual(polygon_area(v), 0.0)

    def test_torch_input(self):
        v = coracpp.zonotope_vertices(torch.zeros(2, requires_grad=True), torch.eye(2))
        self.assertEqual(v.shape, (4, 2))


class Colors(unittest.TestCase):
    """The values of MATLAB CORA's CORAcolor."""

    def test_the_special_colors(self):
        self.assertEqual(coracpp.CORAcolor("CORA:initialSet"), (1.0, 1.0, 1.0))
        self.assertEqual(coracpp.CORAcolor("CORA:simulations"), (0.0, 0.0, 0.0))
        self.assertEqual(coracpp.CORAcolor("CORA:unsafe"), (0.9451, 0.5529, 0.5686))
        self.assertEqual(coracpp.CORAcolor("CORA:unsafeLight"), (0.9059, 0.7373, 0.7373))
        self.assertEqual(coracpp.CORAcolor("CORA:safe"), (0.4706, 0.7725, 0.4980))
        self.assertEqual(coracpp.CORAcolor("CORA:invariant"), coracpp.CORAcolor("CORA:safe"))
        self.assertEqual(coracpp.CORAcolor("CORA:highlight1"), (1.0, 0.6824, 0.2980))
        self.assertEqual(coracpp.CORAcolor("CORA:finalSet"), (0.9, 0.9, 0.9))

    def test_the_palette(self):
        self.assertEqual(coracpp.CORAcolor("CORA:blue"), (0.0, 0.4470, 0.7410))
        self.assertEqual(coracpp.CORAcolor("CORA:color2"), coracpp.CORAcolor("CORA:red"))
        self.assertEqual(coracpp.CORAcolor(3), coracpp.CORAcolor("CORA:yellow"))
        self.assertEqual(coracpp.CORAcolor("CORA:dark-red"), (0.6350, 0.0780, 0.1840))
        self.assertEqual(coracpp.CORAcolor("CORA:color7"), coracpp.CORAcolor("CORA:dark-red"))

    def test_reach_set_shades(self):
        main, worse = (0.2706, 0.5882, 1.0), (0.6902, 0.8235, 1.0)
        self.assertEqual(coracpp.CORAcolor("CORA:reachSet"), main)
        self.assertEqual(coracpp.CORAcolor("CORA:reachSet", 3, 3), main)
        self.assertEqual(coracpp.CORAcolor("CORA:reachSet", 3, 1), worse)
        np.testing.assert_allclose(coracpp.CORAcolor("CORA:reachSet", 3, 2), (np.array(main) + worse) / 2)
        with self.assertRaises(ValueError):
            coracpp.CORAcolor("CORA:reachSet", 2, 3)

    def test_light_and_dark_variants(self):
        base = np.array(coracpp.CORAcolor("CORA:blue"))
        np.testing.assert_allclose(coracpp.CORAcolor("CORA:blue:light"), 0.8 + 0.2 * base)
        np.testing.assert_allclose(coracpp.CORAcolor("CORA:blue:dark"), 0.2 * base)
        np.testing.assert_allclose(coracpp.CORAcolor("CORA:blue:light", alpha=0.5), 0.5 + 0.5 * base)

    def test_unknown_colors(self):
        for bad in ("CORA:mauve", "red", "CORA", "CORA:blue:bright"):
            with self.assertRaises(ValueError):
                coracpp.CORAcolor(bad)


class Plot(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close(self.fig)

    def test_plot_draws_a_reachable_set(self):
        A, c, G = oscillator()
        R = coracpp.reach(A, c, G, time_step=0.1, t_final=1.0)
        coracpp.plot(R, ax=self.ax)
        self.assertEqual(len(self.ax.patches), 10)
        self.assertEqual(tuple(self.ax.patches[0].get_facecolor()[:3]), coracpp.CORAcolor("CORA:reachSet"))

    def test_plot_uses_the_current_axes(self):
        plt.sca(self.ax)
        coracpp.plot(([0.0, 0.0], np.eye(2)))
        self.assertEqual(len(self.ax.patches), 1)

    def test_plot_draws_a_zonotope_simulations_and_points(self):
        A, c, G = oscillator()
        coracpp.plot((c, G), ax=self.ax)
        simulation = coracpp.simulate(A, coracpp.rand_point(c, G, 6, seed=0), 0.1, 1.0)
        coracpp.plot(simulation, ax=self.ax)
        black = [ln for ln in self.ax.lines if ln.get_color() == coracpp.CORAcolor("CORA:simulations")]
        self.assertEqual(len(black), 6 + 1)  # a line per trajectory, and the start dots
        coracpp.plot(coracpp.rand_point(c, G, 30, seed=1), ax=self.ax)
        self.assertEqual(len(self.ax.lines), 6 + 1 + 1)

    def test_plot_draws_a_specification(self):
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        x1 = torch.tensor([1.0, 0.0], **F64)
        # Safe x1 <= 0.5 forbids x1 > 0.5; unsafe x1 <= 0.5 forbids x1 <= 0.5.
        (safe,) = coracpp.plot(coracpp.Specification.safe_set(x1, 0.5), ax=self.ax)
        (unsafe,) = coracpp.plot(coracpp.Specification.unsafe_set(x1, 0.5), ax=self.ax)
        self.assertAlmostEqual(safe.get_xy()[:, 0].min(), 0.5)
        self.assertAlmostEqual(unsafe.get_xy()[:, 0].max(), 0.5)
        self.assertEqual(tuple(safe.get_facecolor()[:3]), coracpp.CORAcolor("CORA:unsafeLight"))

    def test_a_polytope_specification_forbids_each_side(self):
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        x1 = torch.tensor([1.0, 0.0], **F64)
        band = coracpp.Specification.safe_set(x1, 1.0)
        self.assertEqual(band.type, "safeSet")
        (a, b), = band.halfspaces
        np.testing.assert_allclose(a, [1.0, 0.0])
        self.assertEqual(b, 1.0)
        self.assertEqual(coracpp.Specification.unsafe_set(x1, 0.0).type, "unsafeSet")

    def test_labels_reach_the_legend(self):
        A, c, G = oscillator()
        coracpp.plot(coracpp.reach(A, c, G, time_step=0.1, t_final=1.0), ax=self.ax, label="reachable set")
        coracpp.plot_initial_set(c, G, ax=self.ax, label="initial set")
        coracpp.plot(coracpp.simulate(A, coracpp.rand_point(c, G, 3, seed=0), 0.1, 1.0), ax=self.ax, label="simulations")
        _, labels = self.ax.get_legend_handles_labels()
        self.assertEqual(sorted(labels), ["initial set", "reachable set", "simulations"])

    def test_initial_set_is_white_with_a_black_outline(self):
        patch = coracpp.plot_initial_set([0.0, 0.0], np.eye(2), ax=self.ax)
        self.assertEqual(tuple(patch.get_facecolor()[:3]), (1.0, 1.0, 1.0))
        self.assertEqual(tuple(patch.get_edgecolor()[:3]), (0.0, 0.0, 0.0))

    def test_interval(self):
        patch = coracpp.plot_interval([0.0, 1.0], [2.0, 4.0], ax=self.ax)
        self.assertAlmostEqual(polygon_area(patch.get_xy()[:4]), 2.0 * 3.0)

    def test_projection_onto_two_dimensions(self):
        patch = coracpp.plot_zonotope([0.0, 0.0, 5.0], np.eye(3), dims=(0, 2), ax=self.ax)
        self.assertAlmostEqual(patch.get_xy()[:, 1].max(), 6.0)

    def test_an_unknown_object(self):
        with self.assertRaises(TypeError):
            coracpp.plot(3.0, ax=self.ax)


class Examples(unittest.TestCase):
    """Every example runs, so none of them rots."""

    def test_every_python_example_runs(self):
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join(sys.path))
        scripts = sorted((ROOT / "examples" / "python").glob("example_*.py"))
        self.assertGreater(len(scripts), 5)
        with tempfile.TemporaryDirectory() as tmp:
            for script in scripts:
                args = ["--save", os.path.join(tmp, "figure.png")] if script.stem.endswith("_plot") else []
                done = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, env=env)
                self.assertEqual(done.returncode, 0, f"{script.name}: {done.stderr}")
            self.assertGreater(os.path.getsize(os.path.join(tmp, "figure.png")), 10_000)


if __name__ == "__main__":
    unittest.main()
