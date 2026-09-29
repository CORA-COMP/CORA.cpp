"""The plotting helper and the Python example."""
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plotting  # noqa: E402


def polygon_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex_counter_clockwise(v):
    edges = np.roll(v, -1, axis=0) - v
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    return bool(np.all(cross >= -1e-12))


class ZonotopeVertices(unittest.TestCase):
    def test_a_box(self):
        v = plotting.zonotope_vertices([1.0, 2.0], [[1.0, 0.0], [0.0, 3.0]])
        self.assertEqual({tuple(p) for p in v.round(12)}, {(0.0, -1.0), (2.0, -1.0), (2.0, 5.0), (0.0, 5.0)})

    def test_random_zonotopes(self):
        rng = np.random.default_rng(0)
        for m in (1, 2, 3, 5, 8):
            c, G = rng.normal(size=2), rng.normal(size=(2, m))
            v = plotting.zonotope_vertices(c, G)
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
        self.assertEqual(plotting.zonotope_vertices([1.0, 1.0], np.zeros((2, 2))).shape, (1, 2))
        v = plotting.zonotope_vertices([0.0, 0.0], [[1.0, 2.0], [0.0, 0.0]])  # a segment
        self.assertAlmostEqual(polygon_area(v), 0.0)

    def test_torch_input(self):
        import torch
        v = plotting.zonotope_vertices(torch.zeros(2, requires_grad=True), torch.eye(2))
        self.assertEqual(v.shape, (4, 2))


class Plots(unittest.TestCase):
    def test_zonotope_and_halfspace(self):
        fig, ax = plt.subplots()
        patch = plotting.plot_zonotope(ax, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], facecolor="tab:blue")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plotting.plot_halfspace(ax, [1.0, 0.0], 0.5, kind="safe")
        self.assertEqual(len(ax.patches), 2)
        # The shaded side of x1 <= 0.5 is x1 > 0.5: a rectangle 1.5 wide.
        shade = ax.patches[1].get_xy()
        self.assertAlmostEqual(shade[:, 0].min(), 0.5)
        plt.close(fig)

    def test_projection_onto_two_dimensions(self):
        fig, ax = plt.subplots()
        patch = plotting.plot_zonotope(ax, [0.0, 0.0, 5.0], np.eye(3), dims=(0, 2))
        self.assertAlmostEqual(patch.get_xy()[:, 1].max(), 6.0)
        plt.close(fig)


class Example(unittest.TestCase):
    def test_the_example_runs(self):
        try:
            import coracpp  # noqa: F401
        except ImportError:
            self.skipTest("the coracpp module is not built")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "reach.png")
            done = subprocess.run([sys.executable, str(ROOT / "examples" / "python" / "linear_sys.py"), "--save", out],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertGreater(os.path.getsize(out), 10_000)
            self.assertIn("Eigen and libtorch differ by", done.stdout)


if __name__ == "__main__":
    unittest.main()
