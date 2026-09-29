"""example_linear_reach_05_batch - reachability of a batch of initial sets in one call

A batch lives in the object: Zonotope.stack joins four sets into one (c of shape (4, 2), G of
shape (4, 2, 2)), and the call is the one of a single set. libtorch broadcasts the leading dimensions
of the set and of the system, so LinearSys takes a batch of systems the same way.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_05_batch.py
Outputs:  the shape of the result, and whether each member equals its own single run
See also: example_linear_reach_01_5dim, example_linear_reach_06_gpu
"""

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

tFinal = 3.0

# Four initial sets: different centers, and boxes of different size.
centers = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -2.0]]
scales = [0.05, 0.1, 0.2, 0.4]
singles = [cora.Zonotope(cora.Tensor(c), s * cora.eye(2)) for c, s in zip(centers, scales)]
batchR0 = cora.Zonotope.stack(singles)

# Reachability Settings ---------------------------------------------------------------------

timeStep = 0.1
taylorTerms = 8

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))  # one system for all

# Reachability Analysis ---------------------------------------------------------------------

R = sys.reach(batchR0, timeStep, tFinal, taylorTerms)

# Evaluation --------------------------------------------------------------------------------

# Every set of the result carries the batch: c has shape (4, 2), G (4, 2, m).
last = R.timeInt[-1].c
print("sets:", len(singles), " last centers:", tuple(last.shape))
print("distance from the origin at t = 3, by set:", last.norm(dim=-1).tolist())

# Verification ------------------------------------------------------------------------------

# Each member of the batch is what the set gives on its own.
for b, single in enumerate(singles):
    Rb = sys.reach(single, timeStep, tFinal, taylorTerms)
    gap = (last[b] - Rb.timeInt[-1].c).abs().max().item()
    print(f"set {b} differs from its single run by {gap:.1e}")

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
