"""Plotting, as in CORA: coracpp.plot draws what it is given — a reachable set, the initial set,
simulations, a specification — in CORA's colors.

    PYTHONPATH=build python examples/python/example_linearSys_plot.py [--save reach.png]
"""
import argparse

import torch

import coracpp

parser = argparse.ArgumentParser()
parser.add_argument("--save", help="write the figure here instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

f64 = dict(dtype=torch.float64)
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **f64)
c = torch.tensor([1.0, 0.0], **f64)
G = 0.1 * torch.eye(2, **f64)

R = coracpp.reach(A, c, G, time_step=0.1, t_final=6.0, taylor_terms=8)
simulation = coracpp.simulate(A, coracpp.rand_point(c, G, 20, seed=1), 0.05, 6.0)
wall = coracpp.Specification.unsafe_set(torch.tensor([1.0, 0.0], **f64), -0.75)

fig, ax = plt.subplots(figsize=(5, 5))
coracpp.plot(R, label="reachable set")  # dims=(0, 1) by default, as CORA's [1 2]
coracpp.plot_initial_set(c, G, label="initial set")
coracpp.plot(simulation, label="simulations")
coracpp.plot(wall, label="unsafe set")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc="lower left")
if args.save:
    fig.savefig(args.save, dpi=130, bbox_inches="tight")
    print("wrote", args.save)
else:
    plt.show()
