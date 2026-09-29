"""Specifications: check that a reachable set stays in a halfspace (safe set) or out of one
(unsafe set), and find the step that first breaks it.

    PYTHONPATH=build python examples/python/example_linearSys_specification.py
"""
import torch

import coracpp

f64 = dict(dtype=torch.float64)
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **f64)
c = torch.tensor([1.0, 0.0], **f64)
G = 0.1 * torch.eye(2, **f64)
R = coracpp.reach(A, c, G, timeStep=0.1, tFinal=6.0)

# A halfspace is {x | a.x <= b}. Safe: the set must stay inside; here x1 <= 1.2.
stay_left = coracpp.Specification.safeSet(torch.tensor([1.0, 0.0], **f64), 1.2)
print("stays in x1 <= 1.2:", stay_left.check(R))

# Unsafe: the set must not touch it; here x1 <= -0.75, a wall on the left.
wall = coracpp.Specification.unsafeSet(torch.tensor([1.0, 0.0], **f64), -0.75)
print("avoids x1 <= -0.75:", wall.check(R), " first touches it in step", wall.firstViolation(R))
