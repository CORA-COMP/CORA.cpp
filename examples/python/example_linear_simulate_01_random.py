"""example_linear_simulate_01_random - simulations from random points of the initial set

Random points of the initial set, run through the dynamics, must stay inside the reachable set.
Extreme points (corners of the generator cube) are where a set that is too small shows first.

Syntax:   PYTHONPATH=build python examples/python/example_linear_simulate_01_random.py
Outputs:  the size of the simulation, and whether it stays in the reachable set
See also: example_linear_reach_01_5dim
"""

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

tFinal = 1.0
R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))
rng = cora.Rng(1)  # a seed makes the points repeatable

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))

# Reachability Analysis ---------------------------------------------------------------------

timeStep = 0.1
R = sys.reach(R0, timeStep, tFinal, taylorTerms=8)

# Simulation --------------------------------------------------------------------------------

# traj[k] holds the 20 points at time k * timeStep / 2: (time points, n, points).
traj = sys.simulateRandom(R0, 20, timeStep / 2, tFinal, rng)

# The corners of the generator cube as start points, and their trajectories.
corners = R0.randPoint(20, rng, type="extreme")
trajExtreme = sys.simulate(corners, timeStep / 2, tFinal)

# Verification ------------------------------------------------------------------------------

# A point is in the reachable set of its step if d'x is at most the support function there.
d = cora.Tensor([-1.0, 0.5])
inside = True
for j in range(trajExtreme.shape[0]):
    step = min(j // 2, len(R.timeInt) - 1)  # two points per step
    inside &= bool((d @ trajExtreme[j]).max() <= R.timeInt[step].supportFunc(d) + 1e-9)
print("simulation:", tuple(traj.shape), "(time points, n, points)")
print("trajectories from the corners stay in the reachable set:", inside)

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
