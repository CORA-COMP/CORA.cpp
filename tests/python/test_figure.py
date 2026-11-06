"""test_figure - cora.subplots: a figure of several axes that remembers what CORA drew on each

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import os
import tempfile
import unittest

import matplotlib


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

matplotlib.use("Agg")
import cora
import matplotlib.pyplot as plt


# ============================================  MAIN  ============================================ #

class Subplots(unittest.TestCase):
    def setUp(self):
        cora.use_cora_colors("CORA:default")
        self.X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))
        self.sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))

    def tearDown(self):
        plt.close("all")

    def test_it_wraps_the_axes_in_rows_and_columns(self):
        fig = cora.subplots(2, 3)
        self.assertEqual((fig.rows, fig.cols, len(fig)), (2, 3, 6))
        self.assertIs(fig[1, 2], fig.axes[5])
        self.assertIs(fig[4], fig.axes[4])
        self.assertEqual(list(fig), fig.axes)
        self.assertIsInstance(fig.mpl, matplotlib.figure.Figure)
        self.assertEqual(len(fig.mpl.axes), 6)

    def test_what_is_drawn_stays_with_its_axes(self):
        fig = cora.subplots(1, 2)
        cora.plot(self.X0, ax=fig[0, 0])
        cora.plot(self.X0, ax=fig[0, 0])
        cora.plot(self.X0, ax=fig[0, 1])
        # cells hold the patches
        self.assertEqual(len(fig[0, 0].patches), 2)
        grid = fig.core()
        self.assertEqual([len(grid.at(0, c).layers()) for c in range(2)], [2, 1])

    def test_axes_without_a_figure_are_drawn_in_cora_colors_too(self):
        _, ax = plt.subplots()
        cora.plot(self.X0, ax=ax)
        self.assertEqual(ax._cora_figure.layers()[0]["face"], cora.cora_color("CORA:blue"))

    def test_sharing_is_for_matplotlib_and_the_core(self):
        fig = cora.subplots(2, 1, sharex=True)
        cora.plot(self.X0, ax=fig[0, 0])
        cora.plot(self.X0, ax=fig[1, 0])
        self.assertIs(fig[0, 0].get_shared_x_axes().joined(fig[0, 0], fig[1, 0]), True)
        self.assertTrue(fig.core().share_x)
        self.assertFalse(fig.core().share_y)

    def test_a_cell_can_be_three_dimensional(self):
        fig = cora.subplots(1, 2, projection={(0, 1): "3d"})
        self.assertNotEqual(fig[0, 0].name, "3d")
        self.assertEqual(fig[0, 1].name, "3d")
        Z = cora.Zonotope(cora.zeros(3), cora.eye(3))
        cora.plot(Z, dims=(0, 1, 2), ax=fig[0, 1])
        # a 3-D cell
        self.assertEqual(fig.core().at(0, 1).dimension, 3)
        self.assertEqual(cora.subplots(1, 1, projection="3d")[0, 0].name, "3d")

    def test_the_figure_is_svg_in_a_notebook(self):
        fig = cora.subplots(1, 2)
        cora.plot(self.X0, ax=fig[0, 0], label="initial set")
        svg = fig._repr_svg_()
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg, fig.svg())
        self.assertIn("initial set", svg)

    def test_the_texts_and_limits_of_matplotlib_are_taken_over(self):
        fig = cora.subplots(1, 2)
        cora.plot(self.X0, ax=fig[0, 0])
        fig[0, 0].set_title("phase")
        fig[0, 0].set_xlabel("$x_1$")
        fig[0, 0].set_xlim(-1, 3)
        grid = fig.core()
        cell = grid.at(0, 0)
        self.assertEqual((cell.title, cell.xlabel), ("phase", "$x_1$"))
        self.assertEqual(tuple(cell.xlim), (-1.0, 3.0))
        self.assertIsNone(grid.at(0, 1).xlim)

    # a matplotlib figure becomes a CORA figure
    def test_a_matplotlib_figure_becomes_a_cora_figure(self):
        mpl_figure, axes = plt.subplots(2, 2)
        cora.plot(self.X0, ax=axes[1, 0])
        fig = cora.Figure.of(mpl_figure)
        self.assertEqual((fig.rows, fig.cols), (2, 2))
        self.assertEqual(len(fig.core().at(1, 0).layers()), 1)
        text = fig.tikz()
        self.assertIn("group size=2 by 2", text)

    def test_the_figure_is_saved_like_a_matplotlib_one(self):
        fig = cora.subplots(1, 1)
        cora.plot(self.X0, ax=fig[0, 0])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "figure.png")
            fig.savefig(path)
            self.assertGreater(os.path.getsize(path), 0)
            fig.to_tikz(os.path.join(folder, "figure.tikz"))
            self.assertTrue(os.path.exists(os.path.join(folder, "figure.tikz")))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
