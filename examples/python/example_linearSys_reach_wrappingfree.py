"""The two algorithms: "standard" encloses every step's set again, "wrapping-free" computes
the first step's enclosure once and maps it forward. A fast rotation with big steps, from an
initial set that is not aligned with the axes, shows the difference: "standard" is tighter.

    PYTHONPATH=build python examples/python/example_linearSys_reach_wrappingfree.py
"""
import torch

import coracpp

A = torch.tensor([[-0.1, 3.0], [-3.0, -0.1]], dtype=torch.float64)
c = torch.tensor([1.0, 0.5], dtype=torch.float64)
G = torch.tensor([[0.15, 0.05], [-0.05, 0.1]], dtype=torch.float64)

standard = coracpp.reach(A, c, G, time_step=0.3, t_final=2.4, algorithm="standard")
wrapping_free = coracpp.reach(A, c, G, time_step=0.3, t_final=2.4, algorithm="wrapping-free")


def extent_x1(R):
    """How far each enclosure reaches along x1: c_1 + the sum of |G_1j|."""
    return R.time_int_c[:, 0] + R.time_int_G[:, 0].abs().sum(-1)


print("step   standard   wrapping-free")
for k, (a, b) in enumerate(zip(extent_x1(standard).tolist(), extent_x1(wrapping_free).tolist())):
    print(f"{k:4d}   {a:8.5f}   {b:8.5f}")
