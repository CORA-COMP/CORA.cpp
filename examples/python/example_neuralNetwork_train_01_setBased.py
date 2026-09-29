"""example_neuralNetwork_train_01_setBased - set-based training pushes decision boundaries off the samples

The loss of set-based training (Koller, Ladner, Althoff: "Set-Based Training for Neural Network
Verification", TMLR 2025) combines the usual loss of the center of the output set with the size of
the output set, both for the input set of an eps-ball around every training point:

    L = (1 - tau) CE(center of Y, t)  +  tau / eps * ||Y||_F,     ||Y||_F = sqrt(sum G^2) / n

with the output zonotope Y = <c, G>. Here the gradients come from autograd through the set
propagation of `NeuralNetwork.evaluate`, which is differentiable. As in the paper's illustration, a
small set of points with random classes is learned perfectly by a standard and by a set-based model;
the standard model's decision boundary cuts through the eps-boxes around the points, the set-based
model's does not, and the boxes it keeps clear are certified by the zonotope propagation.

Syntax:   PYTHONPATH=build python examples/python/example_neuralNetwork_train_01_setBased.py [--save FILE]
Outputs:  accuracy and certified share of the training points for both models, and the figure
See also: example_neuralNetwork_verify_01, example_linear_learn_01_dynamics
"""
import argparse

import torch
from matplotlib.patches import Rectangle
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

numPoints = 40
epsTrain = 0.03       # half-width of the input boxes during set-based training
tau = 0.003           # weight of the output-set size against the loss of the center
steps = 2000
epsilons = [0.01, 0.02, 0.03]
dtype = torch.float64

# Data --------------------------------------------------------------------------------------


def sample(n, gap, seed):
    """n points of the unit square with random classes; points of different classes are more than
    `gap` apart in every norm, so that boxes around them can be kept clear of each other."""
    generator = torch.Generator().manual_seed(seed)
    points, classes = [], []
    while len(points) < n:
        p = torch.rand(2, dtype=dtype, generator=generator)
        c = int(torch.rand(1, generator=generator) > 0.5)
        if all(c == k or (p - q).abs().max() > gap for q, k in zip(points, classes)):
            points.append(p)
            classes.append(c)
    return torch.stack(points), torch.tensor(classes)


points, labels = sample(numPoints, gap=2.5 * epsTrain, seed=3)

# Models and Set-Based Loss -----------------------------------------------------------------


def newModel():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                         nn.Linear(32, 2)).to(dtype)


def outputSet(model, x, eps):
    """The output zonotopes of the eps-boxes around all points x: one batch, one call."""
    net = cora.NeuralNetwork([(m.weight, m.bias) for m in model if isinstance(m, nn.Linear)])
    return net.evaluate(cora.Zonotope(x, eps * torch.eye(2, dtype=dtype).expand(len(x), 2, 2)))


def fRadius(Y):
    """The F-radius of the output sets (Koller et al., Prop. 5): sqrt(sum of squared generators) / n."""
    return ((Y.G ** 2).sum(dim=(-2, -1)) + 1e-12).sqrt() / Y.G.shape[-2]


def certified(model, eps):
    """Per training point whether every input of the eps-box is classified as its class: the largest
    value of (logit of the other class - logit of the true class) over the output set is below 0."""
    d = torch.where((labels == 0)[:, None], torch.tensor([-1.0, 1.0], dtype=dtype),
                    torch.tensor([1.0, -1.0], dtype=dtype))
    return outputSet(model, points, eps).supportFunc(d) < 0


# Training ----------------------------------------------------------------------------------


def train(setBased):
    model = newModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(steps):
        optimizer.zero_grad()
        if setBased:
            # eps grows over the first half of training, so that the network first fits the points
            eps = epsTrain * (0.1 + 0.9 * min(1.0, step / (0.5 * steps)))
            Y = outputSet(model, points, eps)
            loss = (1 - tau) * nn.functional.cross_entropy(Y.c, labels) + tau / eps * fRadius(Y).mean()
        else:
            loss = nn.functional.cross_entropy(model(points), labels)
        loss.backward()
        optimizer.step()
    return model


models = {"standard": train(False), "set-based": train(True)}

# Evaluation --------------------------------------------------------------------------------

print(f"{'method':<10} {'accuracy':>8} " + " ".join(f"{'eps=' + str(e):>9}" for e in epsilons)
      + "   (share of training points whose eps-box is certified)")
for name, model in models.items():
    accuracy = (model(points).argmax(dim=1) == labels).double().mean().item()
    cert = [certified(model, e).double().mean().item() for e in epsilons]
    print(f"{name:<10} {accuracy:8.3f} " + " ".join(f"{c:9.3f}" for c in cert))

# Visualization -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axis = torch.linspace(0, 1, 300, dtype=dtype)
grid = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=-1).reshape(-1, 2)
regions = {name: model(grid).argmax(dim=1).reshape(300, 300) for name, model in models.items()}
colors = ["tab:blue", "tab:orange"]
for ax, (name, model) in zip(axes, models.items()):
    ax.contourf(axis, axis, regions[name], levels=[-0.5, 0.5, 1.5], colors=["#a6c8f5", "#f8dca0"])
    if name == "set-based":  # the boundary of the standard model, for comparison
        ax.contour(axis, axis, regions["standard"].double(), levels=[0.5], colors="k",
                   linestyles="--", linewidths=1)
    for p, c in zip(points, labels):
        ax.add_patch(Rectangle((p[0] - epsTrain, p[1] - epsTrain), 2 * epsTrain, 2 * epsTrain,
                               fill=False, edgecolor="k", linewidth=0.8))
        ax.scatter(p[0], p[1], s=10, color=colors[c], zorder=3)
    ax.set_title(f"{name} training" + (" (dashed: standard)" if name == "set-based" else ""))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

if args.save:
    plt.savefig(args.save, dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
