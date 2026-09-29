"""example_linear_reach_01_5dim - reachability of a five-dimensional linear system

The system of CORA's example of the same name, without its uncertain inputs (not implemented
yet): the reachable set, simulations, and two projections of both.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_01_5dim.py [--save PREFIX]
Outputs:  the run time and two figures (PREFIX_1.png, PREFIX_2.png with --save)
See also: example_linear_reach_02_algorithms
"""
import argparse
import time

import cora

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="PREFIX", help="write the figures instead of showing them")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

tFinal = 5
R0 = cora.Zonotope(cora.ones(5), 0.1 * cora.eye(5))

# Reachability Settings ---------------------------------------------------------------------

timeStep = 0.02
taylorTerms = 4

# System Dynamics ---------------------------------------------------------------------------

A = cora.Tensor([[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0],
                  [0, 0, 0, 0, -2]])
fiveDimSys = cora.LinearSys(A)

# Reachability Analysis ---------------------------------------------------------------------

timerVal = time.perf_counter()
R = fiveDimSys.reach(R0, timeStep, tFinal, taylorTerms)
tComp = time.perf_counter() - timerVal

print(f"computation time of reachable set: {tComp:.3f} s")

# Simulation --------------------------------------------------------------------------------

rng = cora.Rng(0)
traj = fiveDimSys.simulateRandom(R0, 25, timeStep, tFinal, rng)

# Visualization -----------------------------------------------------------------------------

# plot different projections
dims = [(0, 1), (2, 3)]

for k, projDims in enumerate(dims, start=1):
    plt.figure()

    # plot reachable sets
    cora.plot(R, projDims, label="Reachable set")

    # plot initial set
    cora.plot_initial_set(R0, projDims, label="Initial set")

    # plot simulation results
    cora.plot(traj, projDims, label="Simulations")

    # label plot
    plt.xlabel(f"$x_{projDims[0] + 1}$")
    plt.ylabel(f"$x_{projDims[1] + 1}$")
    plt.legend(loc="upper left")
    if args.save:
        plt.savefig(f"{args.save}_{k}.png", dpi=130, bbox_inches="tight")

if not args.save:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
