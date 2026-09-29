"""example_linear_reach_02_algorithms - the two algorithms of reach: standard and wrapping-free

Both cover the same trajectories. "standard" encloses the set of every step again,
"wrapping-free" encloses the first step once and maps it forward. A fast rotation with big steps,
from an initial set that is not aligned with the axes, shows how far apart they are.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_02_algorithms.py
Outputs:  the extent of every enclosure along x1, for both algorithms
See also: example_linear_reach_01_5dim
"""

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

tFinal = 2.4
R0 = cora.Zonotope(cora.Tensor([1.0, 0.5]),
                   cora.Tensor([[0.15, 0.05], [-0.05, 0.1]]))

# Reachability Settings ---------------------------------------------------------------------

timeStep = 0.3
taylorTerms = 8

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 3.0], [-3.0, -0.1]]))

# Reachability Analysis ---------------------------------------------------------------------

Rstandard = sys.reach(R0, timeStep, tFinal, taylorTerms, linAlg="standard")
RwrappingFree = sys.reach(R0, timeStep, tFinal, taylorTerms, linAlg="wrapping-free")

# Evaluation --------------------------------------------------------------------------------

# The extent of an enclosure along x1 is its support function in the direction (1, 0).
x1 = cora.Tensor([1.0, 0.0])
print("step   standard   wrapping-free")
for k, (Zs, Zw) in enumerate(zip(Rstandard.timeInt, RwrappingFree.timeInt)):
    print(f"{k:4d}   {float(Zs.supportFunc(x1)):8.5f}   {float(Zw.supportFunc(x1)):8.5f}")

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
