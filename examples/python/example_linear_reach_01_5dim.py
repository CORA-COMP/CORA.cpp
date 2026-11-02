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

t_final = 5
R0 = cora.Zonotope(cora.ones(5), 0.1 * cora.eye(5))

# Reachability Settings ---------------------------------------------------------------------

time_step = 0.02
taylor_terms = 4

# System Dynamics ---------------------------------------------------------------------------

A = cora.Tensor([[-1, -4, 0, 0, 0], [4, -1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0],
                  [0, 0, 0, 0, -2]])
five_dim_sys = cora.LinearSys(A)

# Reachability Analysis ---------------------------------------------------------------------

timer_val = time.perf_counter()
R = five_dim_sys.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=taylor_terms)
computation_time = time.perf_counter() - timer_val

print(f"computation time of reachable set: {computation_time:.3f} s")

# Simulation --------------------------------------------------------------------------------

rng = cora.Rng(0)
traj = five_dim_sys.simulate_random(R0, points=25, time_step=time_step, t_final=t_final, rng=rng)

# Visualization -----------------------------------------------------------------------------

# plot different projections
dims = [(0, 1), (2, 3)]

cora.use_cora_colors("CORA:contDynamics")
for k, proj_dims in enumerate(dims, start=1):
    plt.figure()

    # plot reachable sets
    cora.plot(R, proj_dims, label="Reachable set")

    # plot initial set
    cora.plot(R0, proj_dims, label="Initial set")

    # plot simulation results
    cora.plot(traj, proj_dims, label="Simulations")

    # label plot
    plt.xlabel(f"$x_{proj_dims[0] + 1}$")
    plt.ylabel(f"$x_{proj_dims[1] + 1}$")
    plt.legend(loc="upper left")
    if args.save:
        plt.savefig(f"{args.save}_{k}.png", dpi=130, bbox_inches="tight")

if not args.save:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
