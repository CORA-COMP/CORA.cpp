"""example_linear_reach_08_inputs - reachability of a linear system with an input and an output

A mass-spring-damper x' = A x + B u, y = C x: a force u(t) in U acts on the mass, the position is
measured. The reachable states, the output sets, and simulations with constant forces.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_08_inputs.py [--save PREFIX]
Outputs:  the run time, the range of the output, and a figure (PREFIX.png with --save)
See also: example_linear_reach_01_5dim, example_linear_simulate_01_random
"""
import argparse
import time

import cora

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="PREFIX", help="write the figure instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

tFinal = 5
R0 = cora.Zonotope(cora.Tensor([0, 0]), 0.2 * cora.eye(2))            # position, velocity
U = cora.Zonotope(cora.Tensor([0.5]), cora.Tensor([[0.25]]))          # force in [0.25, 0.75]

# Reachability Settings ---------------------------------------------------------------------

timeStep = 0.05
taylorTerms = 6
zonotopeOrder = 20

# System Dynamics ---------------------------------------------------------------------------

A = cora.Tensor([[0, 1], [-4, -0.4]])
B = cora.Tensor([[0], [1]])
C = cora.Tensor([[1, 0]])                                              # the position
sys = cora.LinearSys(A, B, C)

# Reachability Analysis ---------------------------------------------------------------------

timerVal = time.perf_counter()
R = sys.reach(R0, timeStep, tFinal, taylorTerms, U=U, zonotopeOrder=zonotopeOrder)
Y = sys.outputSet(R)
tComp = time.perf_counter() - timerVal

print(f"computation time of reachable set: {tComp:.3f} s")

# Simulation --------------------------------------------------------------------------------

rng = cora.Rng(0)
traj = sys.simulateRandom(R0, 25, timeStep, tFinal, rng, U=U)

# Evaluation --------------------------------------------------------------------------------

# the position over all steps, from the output sets
lower = min(float(y.interval().inf[0]) for y in Y.timeInt)
upper = max(float(y.interval().sup[0]) for y in Y.timeInt)
print(f"the output y stays in [{lower:.4f}, {upper:.4f}]")

# Visualization -----------------------------------------------------------------------------

cora.useCORAcolors("CORA:contDynamics")
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
