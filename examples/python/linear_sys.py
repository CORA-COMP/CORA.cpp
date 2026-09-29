"""Reachability of a damped oscillator from Python: both algorithms, a specification, a batch
of systems, a gradient, and a plot.

    make python TORCH=...
    PYTHONPATH=build python examples/python/linear_sys.py [--device cuda] [--save reach.png]

Torch tensors run on libtorch (any device, batched, differentiable); numpy arrays on Eigen.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import coracpp  # noqa: E402  (built into build/)
from plotting import plot_halfspace, plot_reach  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="cpu or cuda")
parser.add_argument("--save", help="write the figure here instead of showing it")
args = parser.parse_args()
dev = dict(dtype=torch.float64, device=args.device)

# x' = A x, a damped oscillator, from a box around (1, 0).
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **dev)
c = torch.tensor([1.0, 0.0], **dev)
G = 0.1 * torch.eye(2, **dev)
options = dict(time_step=0.1, t_final=6.0, taylor_terms=8)

reach = {name: coracpp.reach(A, c, G, algorithm=name, **options)
         for name in ("standard", "wrapping-free")}
print("steps:", reach["standard"].time_int_c.shape[0])

# A specification: never touch the halfspace x1 <= -0.75 (a wall on the left). The oscillator
# swings towards it, so `first_violation` is the step that first touches it, or -1.
wall = torch.tensor([1.0, 0.0], **dev)
spec = coracpp.Specification.unsafe_set(wall, -0.75)
for name, r in reach.items():
    print(f"{name:14s} safe: {spec.check(r)}   first violation: {spec.first_violation(r)}")

# A batch of systems is one call: leading dimensions of A, c and G broadcast.
dampings = torch.tensor([0.05, 0.1, 0.2], **dev)
As = torch.stack([torch.tensor([[-d, 1.0], [-1.0, -d]], **dev) for d in dampings])
batch = coracpp.reach(As, c, G, **options)
print("batch of 3 systems, time_int_c:", tuple(batch.time_int_c.shape))

# Gradients flow through the whole computation: how the width of the enclosures depends on A.
A_grad = A.clone().requires_grad_()
width = coracpp.reach(A_grad, c, G, **options).time_int_G.abs().sum()
width.backward()
print("d width / dA:\n", A_grad.grad.cpu().numpy())

# numpy arrays run the same algorithm on Eigen.
on_eigen = coracpp.reach(A.cpu().numpy(), c.cpu().numpy(), G.cpu().numpy(), **options)
gap = np.abs(on_eigen.time_int_G - reach["standard"].time_int_G.cpu().numpy()).max()
print("Eigen and libtorch differ by", gap)

if args.save:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, (name, r) in zip(axes, reach.items()):
    plot_reach(ax, r, step=2)
    plot_halfspace(ax, wall, -0.75, kind="unsafe")
    ax.set_title(name)
    ax.set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
fig.suptitle("damped oscillator: reachable sets, dotted = time points, red = unsafe")
fig.tight_layout(rect=(0, 0, 1, 0.94))
if args.save:
    fig.savefig(args.save, dpi=130)
    print("wrote", args.save)
else:
    plt.show()
