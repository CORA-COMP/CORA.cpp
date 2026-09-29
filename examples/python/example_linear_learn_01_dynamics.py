"""example_linear_learn_01_dynamics - learn a system matrix whose reachable sets contain measurements

The initial set is known. Noisy measurements of trajectories that start in it come from a ground
truth system that is not: the matrix A of x' = A x is learned so that the reachable set at every
time point contains the measurements, and stays as small as it can. A is a parameter of a torch
module; the loss is computed on the boxes around the time-point sets, and gradients flow through
LinearSys.reach.

Syntax:   PYTHONPATH=build python examples/python/example_linear_learn_01_dynamics.py [--save FILE]
Outputs:  the loss while learning, the learned matrix, and the figure (FILE with --save)
See also: example_linear_reach_01_5dim, example_linear_simulate_01_random
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

# Ground Truth ------------------------------------------------------------------------------

timeStep = 0.1
tFinal = 2.0
X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))

# The system that produced the data, unknown to the learner.
A_true = cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]])

# Twenty noisy trajectories from the initial set: measurements[k] are the points at time k.
torch.manual_seed(0)
x0 = X0.randPoint(20, cora.Rng(0))
truth = cora.LinearSys(A_true).simulate(x0, timeStep, tFinal)
measurements = truth + 0.01 * torch.randn_like(truth)

# Learning ----------------------------------------------------------------------------------


class LearnedSystem(nn.Module):
    """x' = A x with a learnable A; the loss of the reachable sets against the measurements."""

    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(cora.Tensor([[0.0, 0.5], [-0.5, 0.0]]))

    def forward(self, y):
        R = cora.LinearSys(self.A).reach(X0, timeStep, tFinal, taylorTerms=6)
        outside, width = 0.0, 0.0
        for k in range(1, len(R.timePoint)):
            box = R.timePoint[k].interval()
            # How far the measurements are outside the box around the set at time k, and its size.
            outside = outside + (torch.relu(y[k] - box.sup[:, None])
                                 + torch.relu(box.inf[:, None] - y[k])).mean()
            width = width + (box.sup - box.inf).mean()
        return outside / (len(R.timePoint) - 1), width / (len(R.timePoint) - 1)


model = LearnedSystem()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

for step in range(301):
    optimizer.zero_grad()
    outside, width = model(measurements)
    loss = outside + 0.05 * width  # contain the measurements, and be no larger than needed
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step:3d}: outside {outside.item():.4f}, width {width.item():.3f}")

# Evaluation --------------------------------------------------------------------------------

print("learned A:", model.A.detach().numpy().round(2).tolist())
print("true A:   ", A_true.numpy().round(2).tolist())

# Visualization -----------------------------------------------------------------------------

with torch.no_grad():
    R = cora.LinearSys(model.A).reach(X0, timeStep, tFinal, taylorTerms=6)

cora.useCORAcolors("CORA:contDynamics")
plt.figure()
cora.plot(R, label="Learned reachable set")
cora.plot(X0, label="Initial set")
cora.plot(measurements.reshape(-1, 2, 20).permute(1, 0, 2).reshape(2, -1), label="Measurements",
          color="CORA:red")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc="upper left")
if args.save:
    plt.savefig(args.save, dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
