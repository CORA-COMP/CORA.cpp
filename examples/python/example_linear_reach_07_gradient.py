"""example_linear_reach_07_gradient - how the reachable set depends on the system

torch's autograd differentiates through the whole computation. The matrix exponential can use a
hand-written backward pass instead of autograd's own (custom_backward); the gradient is the same.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_07_gradient.py
Outputs:  the total width of the enclosures and its gradient with respect to A
See also: example_linear_reach_05_batch

Authors:       Claude Sonnet 5.5
Written:       29-September-2026
Last update:   ---
Reviewed:      ---
"""

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------


# ============================================  MAIN  ============================================ #

R0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

def total_width(A, custom_backward):
    """The total width of the enclosures for the system A: a scalar to differentiate."""
    R = cora.LinearSys(A, custom_backward=custom_backward).reach(R0, time_step=0.1, t_final=1.0, taylor_terms=8)
    return sum(Z.generators.abs().sum() for Z in R.time_interval)

# System Dynamics ---------------------------------------------------------------------------

A = cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]], requires_grad=True)

# Gradient ----------------------------------------------------------------------------------

width = total_width(A, custom_backward=False)
width.backward()
print("width", width.item(), " d width / dA:\n", A.grad)

# Custom Backward ---------------------------------------------------------------------------

B = A.detach().clone().requires_grad_()
total_width(B, custom_backward=True).backward()
print("with the custom backward, the largest difference:", (A.grad - B.grad).abs().max().item())

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
