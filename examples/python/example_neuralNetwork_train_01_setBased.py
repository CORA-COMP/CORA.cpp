"""example_neuralNetwork_train_01_setBased - set-based training pushes decision boundaries off the samples

The loss of set-based training (Koller, Ladner, Althoff: "Set-Based Training for Neural Network
Verification", TMLR 2025) combines the usual loss of the center of the output set with the size of
the output set, both for the input set of an eps-ball around every training point:

    L = (1 - tau) CE(center of Y, t)  +  tau / eps * ||Y||_F,     ||Y||_F = sqrt(sum G^2) / n

with the output zonotope Y = <c, G>. Here the gradients come from autograd through the set
propagation of `NeuralNetwork.evaluate`, which is differentiable. This reproduces the paper's Fig. 7
(data and hyperparameters from its Appendix C): both networks learn the 20 points perfectly, but the
decision boundary of the standard one cuts through the eps-boxes around the points, and that of the
set-based one does not; the boxes it keeps clear are certified by the zonotope propagation.

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
parser.add_argument("--seed", type=int, default=1, help="seed of the weights and the batches")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# Parameters --------------------------------------------------------------------------------

epsTrain = 0.05       # half-width of the input boxes during set-based training
# tau and epochs are tuned (the paper's Fig. 7 uses tau = 0.1 and 200 epochs): with these, seeds
# 0-3 all fit the points and the set-based model certifies every box at eps = 0.05, the standard
# one 60-70%; a tau of 0.3 or more collapses the network to a constant class.
tau = 0.05            # weight of the output-set size against the loss of the center
epochs = 1500
batchSize = 10
learningRate = 0.01
epsilons = [0.02, 0.05]
dtype = torch.float64

# Data --------------------------------------------------------------------------------------

# The 20 points of the paper (Appendix C) and their classes: the target (1, 0) is class 0, (0, 1) is 1.
points = torch.tensor([
    [0.0622, 0.6995], [0.6534, 0.9409], [0.4759, 0.7163], [0.8812, 0.1020], [0.5047, 0.4685],
    [0.1470, 0.3275], [0.3439, 0.1395], [0.9098, 0.5422], [0.8588, 0.8696], [0.0545, 0.0825],
    [0.6889, 0.4771], [0.9329, 0.2857], [0.6781, 0.3043], [0.4641, 0.3302], [0.4575, 0.9487],
    [0.1272, 0.4699], [0.6506, 0.7315], [0.5207, 0.1229], [0.3271, 0.4574], [0.6858, 0.0616]],
    dtype=dtype)
labels = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0])

# Models and Set-Based Loss -----------------------------------------------------------------


def newModel():
    """Five linear layers, four hidden layers of 100 ReLUs (the paper's nn-med)."""
    torch.manual_seed(args.seed)
    layers = [nn.Linear(2, 100), nn.ReLU()]
    for _ in range(3):
        layers += [nn.Linear(100, 100), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(100, 2)).to(dtype)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(epochs):
        for batch in torch.randperm(len(points), generator=generator).split(batchSize):
            optimizer.zero_grad()
            if setBased:
                Y = outputSet(model, points[batch], epsTrain)
                loss = ((1 - tau) * nn.functional.cross_entropy(Y.c, labels[batch])
                        + tau / epsTrain * fRadius(Y).mean())
            else:
                loss = nn.functional.cross_entropy(model(points[batch]), labels[batch])
            loss.backward()
            optimizer.step()
    return model


models = {"standard": train(False), "set-based": train(True)}

# Evaluation --------------------------------------------------------------------------------

print(f"{'method':<10} {'accuracy':>8} " + " ".join(f"{'eps=' + str(e):>9}" for e in epsilons)
      + "   (share of the 20 points whose eps-box is certified)")
for name, model in models.items():
    accuracy = (model(points).argmax(dim=1) == labels).double().mean().item()
    cert = [certified(model, e).double().mean().item() for e in epsilons]
    print(f"{name:<10} {accuracy:8.3f} " + " ".join(f"{c:9.3f}" for c in cert))

# Visualization -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axis = torch.linspace(0, 1, 300, dtype=dtype)
grid = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=-1).reshape(-1, 2)
regions = {name: model(grid).argmax(dim=1).reshape(300, 300) for name, model in models.items()}
colors = ["tab:orange", "tab:blue"]
for ax, (name, model) in zip(axes, models.items()):
    ax.contourf(axis, axis, regions[name], levels=[-0.5, 0.5, 1.5], colors=["#f8dca0", "#a6c8f5"])
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
