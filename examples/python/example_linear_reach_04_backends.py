"""example_linear_reach_04_backends - the same code on Eigen (numpy) and libtorch (torch)

An object runs on the backend of the arrays it was made from: numpy arrays on Eigen, torch tensors
on libtorch. The calls are the same, and the results come back as the same kind of array.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_04_backends.py
Outputs:  the last center on each backend, and how much the two differ
See also: example_linear_reach_06_gpu
"""
import numpy as np

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def reachOscillator(asArray):
    """The reachable set of a damped oscillator; asArray turns a list into the array type."""
    # Parameters ----------------------------------------------------------------------------

    R0 = cora.Zonotope(asArray([1.0, 0.0]), asArray([[0.1, 0.0], [0.0, 0.1]]))

    # System Dynamics -----------------------------------------------------------------------

    sys = cora.LinearSys(asArray([[-0.1, 1.0], [-1.0, -0.1]]))

    # Reachability Analysis -----------------------------------------------------------------

    return sys.reach(R0, timeStep=0.1, tFinal=1.0, taylorTerms=8)

# Reachability Analysis ---------------------------------------------------------------------

Reigen = reachOscillator(np.array)
Rtorch = reachOscillator(cora.Tensor)

# Evaluation --------------------------------------------------------------------------------

cEigen, cTorch = Reigen.timeInt[-1].c, Rtorch.timeInt[-1].c
print("eigen:", type(cEigen).__name__, cEigen)
print("torch:", type(cTorch).__name__, cTorch.numpy())
print("the two differ by", float(np.abs(cEigen - cTorch.numpy()).max()))

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
