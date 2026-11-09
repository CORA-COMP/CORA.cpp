"""example_linear_reach_03_specification - checking a reachable set against a specification

A safe set must contain the reachable set, an unsafe set must not touch it. Both are sets, here
polytopes {x | A x <= b}: one row per halfspace.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_03_specification.py
Outputs:  whether each specification holds, and the step that first violates it
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

t_final = 6.0
R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

# Reachability Settings ---------------------------------------------------------------------

time_step = 0.1
taylor_terms = 8

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))

# Specifications ----------------------------------------------------------------------------

x1 = [[1.0, 0.0]]

# Stay in x1 <= 1.2.
spec_safe = cora.Specification.safe_set(cora.Polytope(cora.Tensor(x1), cora.Tensor([1.2])))

# Never touch x1 <= -0.75, a wall on the left.
spec_unsafe = cora.Specification.unsafe_set(cora.Polytope(cora.Tensor(x1), cora.Tensor([-0.75])))

# Reachability Analysis ---------------------------------------------------------------------

R = sys.reach(R0, time_step=time_step, t_final=t_final, taylor_terms=taylor_terms)

# Verification ------------------------------------------------------------------------------

print("stays in x1 <= 1.2:", spec_safe.check(R.time_interval))
print("avoids x1 <= -0.75:", spec_unsafe.check(R.time_interval),
      " first touches it in step", spec_unsafe.first_violation(R.time_interval))

# A specification answers for any set, not only reachable ones.
print("the initial set is safe:", spec_safe.check(R0))

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
