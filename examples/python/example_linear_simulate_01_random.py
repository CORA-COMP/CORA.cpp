"""example_linear_simulate_01_random - simulations from random points of the initial set

Random points of the initial set, run through the dynamics, must stay inside the reachable set.
Extreme points (corners of the generator cube) are where a set that is too small shows first.

Syntax:   PYTHONPATH=build python examples/python/example_linear_simulate_01_random.py
Outputs:  the size of the simulation, and whether it stays in the reachable set
See also: example_linear_reach_01_5dim

Authors:       Claude Sonnet 5.5
Written:       29-September-2026
Last update:   ---
Reviewed:      ---
"""

import cora


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------


# ============================================  MAIN  ============================================ #

t_final = 1.0
R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))
rng = cora.Rng(1)  # a seed makes the points repeatable

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))

# Reachability Analysis ---------------------------------------------------------------------

time_step = 0.1
R = sys.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=8)

# Simulation --------------------------------------------------------------------------------

# traj[k] holds the 20 points at time k * time_step / 2: (time points, n, points).
traj = sys.simulate_random(R0, points=20, time_step=time_step / 2, t_final=t_final, rng=rng)

# The corners of the generator cube as start points, and their trajectories.
corners = R0.rand_point(20, rng, type="extreme")
traj_extreme = sys.simulate(corners, time_step=time_step / 2, t_final=t_final)

# Verification ------------------------------------------------------------------------------

# A point is in the reachable set of its step if d'x is at most the support function there.
d = cora.Tensor([-1.0, 0.5])
inside = True
for j in range(traj_extreme.shape[0]):
    step = min(j // 2, len(R.time_interval) - 1)  # two points per step
    inside &= bool((d @ traj_extreme[j]).max() <= R.time_interval[step].support_function(d) + 1e-9)
print("simulation:", tuple(traj.shape), "(time points, n, points)")
print("trajectories from the corners stay in the reachable set:", inside)

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
