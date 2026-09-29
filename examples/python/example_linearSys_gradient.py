"""Automatic differentiation: how the size of the reachable set depends on the system, with
torch autograd through the whole computation.

    PYTHONPATH=build python examples/python/example_linearSys_gradient.py
"""
import torch

import coracpp

f64 = dict(dtype=torch.float64)
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **f64, requires_grad=True)
c = torch.tensor([1.0, 0.0], **f64)
G = 0.1 * torch.eye(2, **f64)

R = coracpp.reach(A, c, G, time_step=0.1, t_final=1.0)
width = R.time_int_G.abs().sum()  # a scalar: the total width of the enclosures
width.backward()
print("width", width.item(), " d width / dA:\n", A.grad)

# The matrix exponential can use a hand-written backward pass instead of autograd's own.
B = A.detach().clone().requires_grad_()
coracpp.reach(B, c, G, time_step=0.1, t_final=1.0, custom_backward=True).time_int_G.abs().sum().backward()
print("with the custom backward, the largest difference:", (A.grad - B.grad).abs().max().item())
