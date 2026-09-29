"""Simulations: random points of the initial set, run through the dynamics, and a check that
they stay inside the reachable set.

    PYTHONPATH=build python examples/python/example_linearSys_simulate.py
"""
import torch

import coracpp

f64 = dict(dtype=torch.float64)
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **f64)
c = torch.tensor([1.0, 0.0], **f64)
G = 0.1 * torch.eye(2, **f64)

# 20 random points of the initial set (a different seed, a different draw), and 20 corners of it.
start = coracpp.rand_point(c, G, 20, seed=1)
corners = coracpp.rand_point(c, G, 20, seed=2, extreme=True)

# The trajectories through e^{A t}: (time points, n, points).
x = coracpp.simulate(A, start, time_step=0.05, t_final=1.0)
print("simulation:", tuple(x.shape))

# Every point lies under the support function of the reachable set of its step.
R = coracpp.reach(A, c, G, time_step=0.1, t_final=1.0)
d = torch.tensor([-1.0, 0.5], **f64)
y = coracpp.simulate(A, corners, time_step=0.05, t_final=1.0)
inside = True
for j in range(y.shape[0]):
    step = min(j // 2, R.time_int_c.shape[0] - 1)  # 0.05 is half a step of 0.1
    bound = d @ R.time_int_c[step] + (d @ R.time_int_G[step]).abs().sum()
    inside &= bool((d @ y[j]).max() <= bound + 1e-9)
print("trajectories from the corners stay in the reachable set:", inside)
