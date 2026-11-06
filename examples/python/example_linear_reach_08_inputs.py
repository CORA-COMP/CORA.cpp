"""example_linear_reach_08_inputs - reachability of a linear system with an input and an output

A mass-spring-damper x' = A x + B u, y = C x: a force u(t) in U acts on the mass, the position is
measured. The reachable states, the output sets, and simulations with constant forces.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_08_inputs.py [--save PREFIX]
Outputs:  the run time, the range of the output, and a figure (PREFIX.png with --save)
See also: example_linear_reach_01_5dim, example_linear_simulate_01_random

Authors:       Claude Sonnet 5.5
Written:       30-September-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import time

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="PREFIX", help="write the figure instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Parameters --------------------------------------------------------------------------------


t_final = 5
R0 = cora.Zonotope(cora.Tensor([0, 0]), 0.2 * cora.eye(2))            # position, velocity
U = cora.Zonotope(cora.Tensor([0.5]), cora.Tensor([[0.25]]))          # force in [0.25, 0.75]

# Reachability Settings ---------------------------------------------------------------------

time_step = 0.05
taylor_terms = 6
zonotope_order = 20

# System Dynamics ---------------------------------------------------------------------------

A = cora.Tensor([[0, 1], [-4, -0.4]])
B = cora.Tensor([[0], [1]])
C = cora.Tensor([[1, 0]])                                              # the position
sys = cora.LinearSys(A, B, C)

# Reachability Analysis ---------------------------------------------------------------------

timer_val = time.perf_counter()
R = sys.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=taylor_terms, input_set=U, zonotope_order=zonotope_order)
Y = sys.output_set(R)
computation_time = time.perf_counter() - timer_val

print(f"computation time of reachable set: {computation_time:.3f} s")

# Simulation --------------------------------------------------------------------------------

rng = cora.Rng(0)
traj = sys.simulate_random(R0, points=25, time_step=time_step, t_final=t_final, rng=rng, input_set=U)

# Evaluation --------------------------------------------------------------------------------

# the position over all steps, from the output sets
lower = min(float(y.interval().inf[0]) for y in Y.time_interval)
upper = max(float(y.interval().sup[0]) for y in Y.time_interval)
print(f"the output y stays in [{lower:.4f}, {upper:.4f}]")

# Visualization -----------------------------------------------------------------------------

cora.use_cora_colors("CORA:contDynamics")
plt.figure()
cora.plot(R, (0, 1), label="Reachable set")
cora.plot(R0, (0, 1), label="Initial set")
cora.plot(traj, (0, 1), label="Simulations")
plt.xlabel("position")
plt.ylabel("velocity")
plt.legend(loc="upper left")
if args.save:
    plt.savefig(f"{args.save}.png", dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
