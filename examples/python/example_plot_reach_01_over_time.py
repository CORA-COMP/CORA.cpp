"""example_plot_reach_01_over_time - a reachable set in the plane and over time

The reachable set of a damped oscillator in the phase plane (one region, the union of the sets of
all steps), one state over time, simulations, and an unsafe set.

Syntax:   PYTHONPATH=build python examples/python/example_plot_reach_01_over_time.py [--save PREFIX]
Outputs:  two figures (PREFIX_plane.png, PREFIX_x1.png with --save)
See also: example_plot_reach_01_over_time of the C++ examples, example_plot_tikz_01_group

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse

from cora import (
    eye,
    LinearSys,
    plot,
    plot_over_time,
    Polytope,
    Rng,
    Specification,
    Tensor,
    use_cora_colors,
    Zonotope,
)


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="PREFIX", help="write the figures instead of showing them")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Parameters --------------------------------------------------------------------------------


t_final, time_step = 6, 0.1
X0 = Zonotope(Tensor([1, 0]), 0.1 * eye(2))
sys = LinearSys(Tensor([[-0.1, 1], [-1, -0.1]]))

# Reachability Analysis ---------------------------------------------------------------------

R = sys.reach(X0, time_step=time_step, t_final=t_final, taylor_terms=6)
x = sys.simulate_random(X0, points=10, time_step=time_step, t_final=t_final, rng=Rng(0))

# Visualization -----------------------------------------------------------------------------

# The phase plane: reachable set, initial set, simulations, then the unsafe set
use_cora_colors("CORA:contDynamics")
plt.figure()
R.plot((0, 1), label="reachable set")
X0.plot((0, 1), label="initial set")
plot(x, (0, 1), label="simulations")
wall = Polytope(Tensor([[-1.0, 0.0]]), Tensor([0.5]))
Specification.unsafe_set(wall).plot((0, 1), label="unsafe set")
plt.legend()

# The first state over time: boxes over the time intervals, and the simulations.
plt.figure()
R.plot_over_time(0, label="reachable set")
plot_over_time(x, 0, time_step=time_step, label="simulations")
plt.legend()

if args.save:
    for number, name in zip(plt.get_fignums(), ("plane", "x1")):
        plt.figure(number).savefig(f"{args.save}_{name}.png", dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
