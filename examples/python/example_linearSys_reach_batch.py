"""Automatic batching: three systems, or three initial sets, are one call. The leading
dimensions of A, c and G broadcast.

    PYTHONPATH=build python examples/python/example_linearSys_reach_batch.py
"""
import torch

import coracpp

f64 = dict(dtype=torch.float64)
damping = torch.tensor([0.05, 0.1, 0.2], **f64)
# A has shape (3, 2, 2): three oscillators that differ in their damping.
A = torch.stack([torch.tensor([[-d, 1.0], [-1.0, -d]], **f64) for d in damping])
c = torch.tensor([1.0, 0.0], **f64)
G = 0.1 * torch.eye(2, **f64)

# Three systems, one initial set.
R = coracpp.reach(A, c, G, time_step=0.1, t_final=3.0)
print("centers (steps, systems, n):", tuple(R.time_int_c.shape))
print("distance from the origin at t = 3, by damping:", R.time_int_c[-1].norm(dim=-1).tolist())

# Three initial sets, one system: give c (3, n) and G (3, n, m).
cs = torch.stack([c, 2 * c, -c])
Gs = G.expand(3, 2, 2)
R = coracpp.reach(A[1], cs, Gs, time_step=0.1, t_final=3.0)
print("three initial sets:", tuple(R.time_int_c.shape))
