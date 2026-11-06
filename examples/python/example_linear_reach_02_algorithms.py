"""example_linear_reach_02_algorithms - the two algorithms of reach: standard and wrapping-free

Both cover the same trajectories. "standard" encloses the set of every step again,
"wrapping_free" encloses the first step once and maps it forward. A fast rotation with big steps,
from an initial set that is not aligned with the axes, shows how far apart they are.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_02_algorithms.py
Outputs:  the extent of every enclosure along x1, for both algorithms
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

t_final = 2.4
R0 = cora.Zonotope(cora.Tensor([1.0, 0.5]),
                   cora.Tensor([[0.15, 0.05], [-0.05, 0.1]]))

# Reachability Settings ---------------------------------------------------------------------

time_step = 0.3
taylor_terms = 8

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 3.0], [-3.0, -0.1]]))

# Reachability Analysis ---------------------------------------------------------------------

Rstandard = sys.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=taylor_terms, alg="standard")
RwrappingFree = sys.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=taylor_terms, alg="wrapping_free")

# Evaluation --------------------------------------------------------------------------------

# The extent of an enclosure along x1 is its support function in the direction (1, 0).
x1 = cora.Tensor([1.0, 0.0])
print("step   standard   wrapping-free")
for k, (Zs, Zw) in enumerate(zip(Rstandard.time_interval, RwrappingFree.time_interval)):
    print(f"{k:4d}   {float(Zs.support_function(x1)):8.5f}   {float(Zw.support_function(x1)):8.5f}")

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
