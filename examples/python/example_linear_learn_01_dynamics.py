"""example_linear_learn_01_dynamics - learn a system matrix whose reachable sets contain measurements

The initial set is known. Noisy measurements of trajectories that start in it come from a ground
truth system that is not: the matrix A of x' = A x is learned so that the reachable set at every
time point contains the measurements, and stays as small as it can. A is a parameter of a torch
module; the loss is computed on the boxes around the time-point sets, and gradients flow through
LinearSys.reach.

Syntax:   PYTHONPATH=build python examples/python/example_linear_learn_01_dynamics.py [--save FILE]
Outputs:  the loss while learning, the learned matrix, and the figure (FILE with --save)
See also: example_linear_reach_01_5dim, example_linear_simulate_01_random

Authors:       Claude Sonnet 5.5
Written:       29-September-2026
Last update:   ---
Reviewed:      ---
"""
import argparse

from cora import eye, LinearSys, plot, Rng, Tensor, use_cora_colors, Zonotope
import torch
from torch import nn


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--save", metavar="FILE", help="write the figure instead of showing it")
args = parser.parse_args()
if args.save:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ground Truth ------------------------------------------------------------------------------


time_step = 0.1
t_final = 2.0
X0 = Zonotope(Tensor([1.0, 0.0]), 0.1 * eye(2))

# The system that produced the data, unknown to the learner.
A_true = Tensor([[-0.1, 1.0], [-1.0, -0.1]])

# Twenty noisy trajectories from the initial set: measurements[k] are the points at time k.
torch.manual_seed(0)
x0 = X0.rand_point(20, Rng(0))  # the simulation takes the points as rows
truth = LinearSys(A_true).simulate(x0, time_step=time_step, t_final=t_final)
measurements = truth + 0.01 * torch.randn_like(truth)

# Learning ----------------------------------------------------------------------------------


class LearnedSystem(nn.Module):
    """x' = A x with a learnable A; the loss of the reachable sets against the measurements."""

    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(Tensor([[0.0, 0.5], [-0.5, 0.0]]))

    def forward(self, y):
        R = LinearSys(self.A).reach(X0, time_step=time_step, t_final=t_final, taylor_terms=6)
        outside, width = 0.0, 0.0
        for k in range(1, len(R.time_point)):
            box = R.time_point[k].interval()
            # How far the measurements are outside the box around the set at time k, and its size.
            outside = outside + (torch.relu(y[k] - box.sup)
                                 + torch.relu(box.inf - y[k])).mean()
            width = width + (box.sup - box.inf).mean()
        return outside / (len(R.time_point) - 1), width / (len(R.time_point) - 1)


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
    R = LinearSys(model.A).reach(X0, time_step=time_step, t_final=t_final, taylor_terms=6)

use_cora_colors("CORA:contDynamics")
plt.figure()
plot(R, label="Learned reachable set")
plot(X0, label="Initial set")
plot(measurements.reshape(-1, 2), label="Measurements",
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
