"""example_linear_reach_03_specification - checking a reachable set against a specification

A safe set must contain the reachable set, an unsafe set must not touch it. Both are halfspaces
{x | a'x <= b}.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_03_specification.py
Outputs:  whether each specification holds, and the step that first violates it
See also: example_linear_reach_01_5dim
"""

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

tFinal = 6.0
R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

# Reachability Settings ---------------------------------------------------------------------

timeStep = 0.1
taylorTerms = 8

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]]))

# Specifications ----------------------------------------------------------------------------

x1 = cora.Tensor([1.0, 0.0])

# Stay in x1 <= 1.2.
specSafe = cora.Specification.safeSet(x1, 1.2)

# Never touch x1 <= -0.75, a wall on the left.
specUnsafe = cora.Specification.unsafeSet(x1, -0.75)

# Reachability Analysis ---------------------------------------------------------------------

R = sys.reach(R0, timeStep, tFinal, taylorTerms)

# Verification ------------------------------------------------------------------------------

print("stays in x1 <= 1.2:", specSafe.check(R.timeInt))
print("avoids x1 <= -0.75:", specUnsafe.check(R.timeInt),
      " first touches it in step", specUnsafe.firstViolation(R.timeInt))

# A specification answers for any set, not only reachable ones.
print("the initial set is safe:", specSafe.check(R0))

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
