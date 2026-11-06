"""example_plot_tikz_01_group - four axes written as one pgfplots group plot

cora.subplots is plt.subplots that remembers what CORA drew on each axes; to_tikz writes it as one
groupplot with shared axes, CORA's colors by their names in cora-macs and thinned coordinates. The
text comes from the C++ core, the same as for its own figures. Without --save the figure is shown.

Syntax:   PYTHONPATH=build python examples/python/example_plot_tikz_01_group.py [--save PREFIX]
Outputs:  PREFIX.tikz (a tikzpicture) and PREFIX_standalone.tikz (a document)
See also: example_plot_tikz_01_group of the C++ examples, example_plot_reach_01_over_time

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
parser.add_argument("--save", metavar="PREFIX", help="write the TikZ files instead of showing")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Reachability Analysis ---------------------------------------------------------------------


X0 = cora.Zonotope(cora.Tensor([1, 0]), 0.1 * cora.eye(2))
sys = cora.LinearSys(cora.Tensor([[-0.1, 1], [-1, -0.1]]))
R = sys.reach(X0, time_step=0.1, t_final=3.0, taylor_terms=6)

# Figures -----------------------------------------------------------------------------------

cora.use_cora_colors("CORA:contDynamics")
fig = cora.subplots(2, 2, sharex=True, projection={(1, 1): "3d"})

# The phase plane in the first axes, the two states over time in the second and third.
cora.plot(R, (0, 1), ax=fig[0, 0], label="reachable set")
cora.plot(X0, (0, 1), ax=fig[0, 0], label="initial set")
fig[0, 0].set_xlabel("$x_1$")
fig[0, 0].set_ylabel("$x_2$")
cora.plot_over_time(R, 0, ax=fig[0, 1])
fig[0, 1].set_ylabel("$x_1$")
cora.plot_over_time(R, 1, ax=fig[1, 0])
fig[1, 0].set_xlabel("$t$")
fig[1, 0].set_ylabel("$x_2$")

# A three-dimensional set in the last one, whose axes were made with projection "3d".
Z = cora.Zonotope(cora.zeros(3), cora.Tensor([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]))
cora.plot(Z, (0, 1, 2), ax=fig[1, 1], face_color="CORA:blue")
fig[1, 1].set_xlabel("$x_1$")
fig[1, 1].set_ylabel("$x_2$")
fig[1, 1].set_zlabel("$x_3$")

# TikZ Export -------------------------------------------------------------------------------

# A picture for \input into a paper that loads cora-macs, and a document that compiles alone.
if args.save:
    fig.to_tikz(f"{args.save}.tikz")
    fig.to_tikz(f"{args.save}_standalone.tikz", standalone=True)
else:
    print(fig.tikz()[:400], "...")
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
