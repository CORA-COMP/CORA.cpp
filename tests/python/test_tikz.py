"""test_tikz - the TikZ export the C++ core writes for Python figures, and that LaTeX reads it

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import matplotlib


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

matplotlib.use("Agg")
import cora
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with open(path) as f:
        return f.read()


def latex():
    """The pdflatex that has pgfplots, or None."""
    extra = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")
    path = shutil.which("pdflatex") or shutil.which("pdflatex", path=extra)
    if path is None:
        return None
    env = dict(os.environ, PATH=os.path.dirname(path) + os.pathsep + os.environ["PATH"])
    found = subprocess.run(["kpsewhich", "pgfplots.sty"], capture_output=True, text=True, env=env)
    return path if found.returncode == 0 and found.stdout.strip() else None


class Square(cora.ContSet):
    """A square with many collinear points on each edge, for the thinning."""

    def _dim(self): return 2
    def _center(self): return cora.Tensor([0.0, 0.0])
    def _interval(self): return cora.Interval(cora.Tensor([0.0, 0.0]), cora.Tensor([1.0, 1.0]))
    def _project(self, dims): return self
    def _vertices(self):
        t = np.linspace(0, 1, 100, endpoint=False)
        z, o = np.zeros_like(t), np.ones_like(t)
        return [np.concatenate([np.stack([t, z], 1), np.stack([o, t], 1), np.stack([1 - t, o], 1),
                                np.stack([z, 1 - t], 1)])]


# ============================================  MAIN  ============================================ #

class Tikz(unittest.TestCase):
    def setUp(self):
        cora.use_cora_colors("CORA:default")
        self.X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))
        self.sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))
        self.R = self.sys.reach(self.X0, time_step=0.1, t_final=1.0)

    def tearDown(self):
        cora.use_cora_colors("CORA:default")
        plt.close("all")

    def one(self):
        fig = cora.subplots(1, 1)
        cora.plot(self.X0, ax=fig[0, 0], label="$X_0$")
        fig[0, 0].set_xlabel("$x_1$")
        fig[0, 0].set_ylabel("$x_2$")
        return fig

    def group(self):
        cora.use_cora_colors("CORA:contDynamics")
        fig = cora.subplots(2, 2, sharex=True, projection={(1, 1): "3d"})
        cora.plot(self.R, ax=fig[0, 0], label="reachable set")
        cora.plot(self.X0, ax=fig[0, 0])
        # a grid of figures
        cora.plot_over_time(self.R, 0, ax=fig[0, 1])
        cora.plot_over_time(self.R, 1, ax=fig[1, 0])
        fig[1, 0].set_xlabel("$t$")
        cora.plot(cora.Zonotope(cora.zeros(3), cora.eye(3)), (0, 1, 2), ax=fig[1, 1])
        return fig

    def test_one_axes_is_one_picture_with_an_addplot_per_set(self):
        text = self.one().tikz()
        self.assertIn("\\begin{axis}", text)
        self.assertNotIn("groupplot", text)
        self.assertEqual(text.count("\\addplot"), 1)
        self.assertIn("xlabel={$x_1$}", text)
        self.assertIn("\\addlegendentry{$X_0$}", text)
        self.assertIn("plot filled=CORAcolorBlue", text)

    def test_several_axes_are_one_group_plot_with_shared_axes(self):
        text = self.group().tikz()
        self.assertIn("group size=2 by 2", text)
        self.assertEqual(text.count("\\nextgroupplot"), 4)
        self.assertIn("x descriptions at=edge bottom", text)
        self.assertNotIn("y descriptions", text)
        self.assertIn("\\addplot3", text)
        # labels of the cells
        self.assertIn("view={-35}{30}", text)
        xlabels = [line for line in text.splitlines() if line.strip().startswith("xlabel")]
        self.assertEqual(xlabels, ["  xlabel={$t$},", "  xlabel={x[0]},"])  # the bottom row only

    def test_the_text_is_the_same_every_time(self):
        fig = self.group()
        self.assertEqual(fig.tikz(), fig.tikz())

    def test_cora_colors_are_named_or_written_out(self):
        fig = self.group()
        named = fig.tikz()
        self.assertIn("CORAcolorReachSet", named)
        self.assertNotIn("rgb,1", named)
        explicit = fig.tikz(coramacs=False)
        self.assertNotIn("CORAcolor", explicit)
        self.assertIn("fill={rgb,1:red,0.2706;green,0.5882;blue,1}", explicit)
        standalone = fig.tikz(standalone=True)
        self.assertIn("\\definecolor{CORAcolorReachSet}{rgb}{0.2706,0.5882,1}", standalone)
        self.assertIn("plot filled/.style", self.one().tikz(standalone=True))
        self.assertNotIn("plot filled/.style", self.one().tikz(standalone=True, coramacs=False))

    def test_standalone_is_a_document_and_the_default_is_not(self):
        fig = self.one()
        self.assertTrue(fig.tikz(standalone=True).startswith("\\documentclass"))
        self.assertTrue(fig.tikz(standalone=True).rstrip().endswith("\\end{document}"))
        self.assertTrue(fig.tikz().startswith("%"))
        # groupplots
        self.assertIn("groupplots", self.group().tikz(standalone=True))

    def test_collinear_points_are_thinned_within_a_tolerance(self):
        fig = cora.subplots(1, 1)
        cora.plot(Square(), ax=fig[0, 0])
        thin, full = fig.tikz(), fig.tikz(thin=False)
        self.assertLess(thin.count("("), 10)
        self.assertGreater(full.count("("), 400)
        self.assertLess(fig.tikz(thin_tolerance=0.5).count("("), 10)

    def test_the_coordinates_can_go_to_tables_next_to_the_file(self):
        fig = self.group()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "reach.tikz")
            fig.to_tikz(path, external_data=True)
            text = read(path)
            tables = sorted(f for f in os.listdir(folder) if f.endswith(".dat"))
            self.assertEqual(len(tables), text.count("table[x=x"))
            self.assertIn(os.path.join(folder, "reach_1.dat"), text)
            self.assertTrue(read(os.path.join(folder, tables[0])).startswith("x y\n"))
            with self.assertRaises(ValueError):
                fig.tikz(external_data=True)

    # a wrong option is described
    def test_a_wrong_option_is_described(self):
        with self.assertRaises(TypeError) as caught:
            self.one().tikz(standalon=True)
        self.assertIn("standalone", str(caught.exception))

    def test_the_figure_of_the_current_axes_is_written_with_to_tikz(self):
        cora.plot(self.X0)
        with tempfile.TemporaryDirectory() as folder:
            cora.to_tikz(os.path.join(folder, "a.tikz"))
            self.assertIn("\\addplot", read(os.path.join(folder, "a.tikz")))

    @unittest.skipIf(latex() is None, "pdflatex with pgfplots is not installed")
    def test_latex_compiles_the_standalone_files(self):
        pdflatex = latex()
        env = dict(os.environ, PATH=os.path.dirname(pdflatex) + os.pathsep + os.environ["PATH"])
        figures = {"one": self.one(), "group": self.group()}
        with tempfile.TemporaryDirectory() as folder:
            for name, fig in figures.items():
                for coramacs in (True, False):
                    base = os.path.join(folder, f"{name}_{coramacs}")
                    fig.to_tikz(base + ".tex", standalone=True, coramacs=coramacs)
                    # compile the output if pdflatex is there
                    done = subprocess.run(
                        [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                         "-output-directory", folder, base + ".tex"],
                        capture_output=True, text=True, env=env, timeout=300)
                    self.assertEqual(done.returncode, 0, done.stdout[-2000:])
                    self.assertTrue(os.path.exists(base + ".pdf"))
            # the tables are read from where they were written
            base = os.path.join(folder, "tables")
            figures["group"].to_tikz(base + ".tex", standalone=True, external_data=True)
            done = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", folder,
                 base + ".tex"], capture_output=True, text=True, env=env, timeout=300)
            self.assertEqual(done.returncode, 0, done.stdout[-2000:])

if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
