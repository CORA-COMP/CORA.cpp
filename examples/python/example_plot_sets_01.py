"""example_plot_sets_01 - a set in one, two and three dimensions

cora.plot draws the same set projected onto one dimension (a bar), two (a polygon) or three (the
faces of a polytope; the axes need projection="3d"). The sets also plot themselves: Z.plot(dims).

Syntax:   PYTHONPATH=build python examples/python/example_plot_sets_01.py [--save PREFIX]
Outputs:  three figures (PREFIX_1d.png, PREFIX_2d.png, PREFIX_3d.png with --save)
See also: example_plot_sets_01 of the C++ examples, example_plot_tikz_01_group

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="PREFIX", help="write the figures instead of showing them")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Sets --------------------------------------------------------------------------------------


Z = cora.Zonotope(cora.zeros(3),
                  cora.Tensor([[1, 0, 0.5, 0.2], [0, 1, 0.5, -0.3], [0, 0, 1, 0.4]]))
I = cora.Interval(cora.Tensor([-1.5, -1, -0.5]), cora.Tensor([0.5, 1.5, 2]))

# One Dimension -----------------------------------------------------------------------------

plt.figure()
Z.plot((0,), label="zonotope", line_width=4)
I.plot((0,), label="interval", line_width=4)
plt.legend()

# Two Dimensions ----------------------------------------------------------------------------

plt.figure()
Z.plot((0, 1), label="zonotope")
I.plot((0, 1), label="interval", filled=False)
plt.legend()
plt.title("two dimensions")

# Three Dimensions --------------------------------------------------------------------------

# The faces come from the convex hull of the vertices, computed once in the C++ core.
ax = plt.figure().add_subplot(projection="3d")
Z.plot((0, 1, 2), ax=ax, label="zonotope")
I.plot((0, 1, 2), ax=ax, label="interval", face_alpha=0.1)

if args.save:
    for number, name in zip(plt.get_fignums(), ("1d", "2d", "3d")):
        plt.figure(number).savefig(f"{args.save}_{name}.png", dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
