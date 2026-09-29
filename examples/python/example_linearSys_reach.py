"""The reachable set of a damped oscillator, with torch tensors (libtorch).

    make python TORCH=...
    PYTHONPATH=build python examples/python/example_linearSys_reach.py
"""
import torch

import coracpp

A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], dtype=torch.float64)
c = torch.tensor([1.0, 0.0], dtype=torch.float64)  # the initial set: center ...
G = 0.1 * torch.eye(2, dtype=torch.float64)  # ... and generators

R = coracpp.reach(A, c, G, time_step=0.1, t_final=1.0, taylor_terms=8)

# The steps are stacked in the first dimension: time_int_* has 10 (one per step), time_point_*
# has 11 (the start and the end of each step).
print("time-interval centers:", tuple(R.time_int_c.shape), " generators:", tuple(R.time_int_G.shape))
print("center of the last enclosure:", R.time_int_c[-1].tolist())
