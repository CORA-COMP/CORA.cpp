"""example_neuralNetwork_verify_01 - certify a classifier against perturbations of its input

A small ReLU network is trained to tell the inside of a circle from the outside. A perturbation of
size eps around a test point is a zonotope; it goes through the network as a zonotope (affine layers
exactly, every ReLU by the tightest parallelogram of one slope), and the classification is certified
if the largest logit of any other class stays below that of the predicted one over the whole output
set. All test points are verified in one call, as a batch.

Syntax:   PYTHONPATH=build python examples/python/example_neuralNetwork_verify_01.py [--save FILE]
Outputs:  the share of certified points per perturbation size, and the figure (FILE with --save)
See also: example_linear_learn_01_dynamics, example_zonotope_01
"""
import argparse

import torch
from torch import nn

import cora

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="FILE", help="write the figure instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

radius = 0.7                       # the circle that separates the classes
epsilons = [0.01, 0.03, 0.06, 0.1]  # half-widths of the perturbation boxes
plotEps = 0.06                     # the perturbation shown in the figure

# Training ----------------------------------------------------------------------------------

torch.manual_seed(0)
dtype = torch.float64


def sample(n):
    """Points in [-1.5, 1.5]^2 and their class: 0 inside the circle, 1 outside."""
    x = 3 * torch.rand(n, 2, dtype=dtype) - 1.5
    return x, (x.norm(dim=1) > radius).long()


model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2)).to(dtype)
x, labels = sample(1500)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(600):
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x), labels)
    loss.backward()
    optimizer.step()

points, truth = sample(400)
predicted = model(points).argmax(dim=1)
print(f"training loss {loss.item():.3f}, test accuracy {(predicted == truth).double().mean().item():.3f}")

# Verification ------------------------------------------------------------------------------

# The network of cora has the weights of the torch model.
net = cora.NeuralNetwork([(m.weight.detach(), m.bias.detach()) for m in model if isinstance(m, nn.Linear)])

# Along d the output logit of the other class minus that of the predicted one: d = (-1, 1) for
# class 0, (1, -1) for class 1. If its largest value over the output set is below 0, the class holds.
d = torch.where((predicted == 0)[:, None], torch.tensor([-1.0, 1.0], dtype=dtype),
                torch.tensor([1.0, -1.0], dtype=dtype))


def perturbations(eps):
    """The boxes of half-width eps around every test point, as one batch of zonotopes."""
    return cora.Zonotope(points, eps * torch.eye(2, dtype=dtype).expand(len(points), 2, 2))


generator = torch.Generator().manual_seed(1)
print(" eps   certified   robust under random perturbations")
for eps in epsilons:
    certified = net.evaluate(perturbations(eps)).supportFunc(d) < 0
    # A point that is certified must stay put: try 200 random perturbations of every point.
    noise = eps * (2 * torch.rand(200, len(points), 2, generator=generator, dtype=dtype) - 1)
    stays = (model(points + noise).argmax(dim=-1) == predicted).all(dim=0)
    assert not bool((certified & ~stays).any()), "a certified point changed its class"
    print(f"{eps:5.2f}   {certified.double().mean().item():8.1%}   {stays.double().mean().item():8.1%}")

# Visualization -----------------------------------------------------------------------------

certified = net.evaluate(perturbations(plotEps)).supportFunc(d) < 0
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))

plt.sca(left)
angle = torch.linspace(0, 2 * torch.pi, 200)
left.plot(radius * torch.cos(angle), radius * torch.sin(angle), "k--", linewidth=1, label="Class boundary")
left.scatter(*points[certified].T, s=14, color="tab:green", label=f"Certified (eps = {plotEps})")
left.scatter(*points[~certified].T, s=14, color="tab:red", label="Not certified")
left.set_aspect("equal")
left.set_xlabel("$x_1$")
left.set_ylabel("$x_2$")
left.legend(loc="upper right", fontsize=8)

# The output sets, in logit space, of the certified and the not certified point closest to the class
# boundary, with sampled outputs; the dashed line is where the logits are equal.
plt.sca(right)
fromBoundary = (points.norm(dim=1) - radius).abs()
lo, hi = torch.full((2,), torch.inf, dtype=dtype), torch.full((2,), -torch.inf, dtype=dtype)
for label, mask in (("certified", certified), ("not certified", ~certified)):
    index = torch.where(mask)[0][fromBoundary[mask].argmin()]
    X = cora.Zonotope(points[index], plotEps * torch.eye(2, dtype=dtype))
    Y = net.evaluate(X)
    cora.plot(Y, label=f"Output set ({label})")
    y = net.evaluate(X.randPoint(300, cora.Rng(0)))
    right.scatter(y[0], y[1], s=4, label=f"Sampled outputs ({label})")
    box = Y.interval()
    lo, hi = torch.minimum(lo, box.inf), torch.maximum(hi, box.sup)
pad = 0.25 * (hi - lo)
right.set_xlim(float(lo[0] - pad[0]), float(hi[0] + pad[0]))
right.set_ylim(float(lo[1] - pad[1]), float(hi[1] + pad[1]))
right.axline((0, 0), slope=1, color="k", linestyle="--", linewidth=1, label="Equal logits")
right.set_xlabel("logit of class 0")
right.set_ylabel("logit of class 1")
right.legend(loc="upper right", fontsize=8)

if args.save:
    plt.savefig(args.save, dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
