"""example_linear_reach_07_gradient - how the reachable set depends on the system

torch's autograd differentiates through the whole computation. The matrix exponential can use a
hand-written backward pass instead of autograd's own (customBackward); the gradient is the same.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_07_gradient.py
Outputs:  the total width of the enclosures and its gradient with respect to A
See also: example_linear_reach_05_batch
"""

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

def totalWidth(A, customBackward):
    """The total width of the enclosures for the system A: a scalar to differentiate."""
    R = cora.LinearSys(A, customBackward=customBackward).reach(R0, 0.1, 1.0, 8)
    return sum(Z.G.abs().sum() for Z in R.timeInt)

# System Dynamics ---------------------------------------------------------------------------

A = cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]], requires_grad=True)

# Gradient ----------------------------------------------------------------------------------

width = totalWidth(A, customBackward=False)
width.backward()
print("width", width.item(), " d width / dA:\n", A.grad)

# Custom Backward ---------------------------------------------------------------------------

B = A.detach().clone().requires_grad_()
totalWidth(B, customBackward=True).backward()
print("with the custom backward, the largest difference:", (A.grad - B.grad).abs().max().item())

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
