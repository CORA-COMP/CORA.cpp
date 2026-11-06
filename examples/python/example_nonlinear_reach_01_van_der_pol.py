"""example_nonlinear_reach_01_van_der_pol - reachability of the van der Pol oscillator

The dynamics is written once on symbolic states; the system differentiates it, and reach
linearizes it step by step (CORA's algorithm "lin"). The reachable set follows the limit cycle
for one period; simulations from the initial set must stay inside it.

Syntax:   PYTHONPATH=build python examples/python/example_nonlinear_reach_01_van_der_pol.py [--save FILE]
Outputs:  the run time and the figure (FILE with --save)
See also: example_linear_reach_01_5dim

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import time

import cora


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="FILE", help="write the figure instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Parameters --------------------------------------------------------------------------------


t_final = 6.74
R0 = cora.Zonotope(cora.Tensor([1.4, 2.3]), 0.05 * cora.eye(2))

# Reachability Settings ---------------------------------------------------------------------

time_step = 0.005
taylor_terms = 4
zonotope_order = 50

# System Dynamics ---------------------------------------------------------------------------

mu = 1
vdp = cora.NonlinearSys(lambda x: [x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]], 2)

# Reachability Analysis ---------------------------------------------------------------------

timer_val = time.perf_counter()
R = vdp.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=taylor_terms, zonotope_order=zonotope_order)
computation_time = time.perf_counter() - timer_val

print(f"computation time of reachable set: {computation_time:.3f} s")

# Simulation --------------------------------------------------------------------------------

rng = cora.Rng(0)
traj = vdp.simulate_random(R0, points=10, time_step=time_step, t_final=t_final, rng=rng)

# Visualization -----------------------------------------------------------------------------

cora.use_cora_colors("CORA:contDynamics")
plt.figure()

# plot reachable sets
cora.plot(R, (0, 1), label="Reachable set")

# plot initial set
cora.plot(R0, (0, 1), label="Initial set")

# plot simulation results
cora.plot(traj, (0, 1), label="Simulations")

# label plot
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc="upper left")
if args.save:
    plt.savefig(args.save, dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
