"""example_linear_learn_02_conformal - a learned linear model with a conformal guarantee on a nonlinear system

Noisy trajectories of a damped pendulum (a nonlinear system) start in a known set. A linear model
x' = A x is learned from some of them, and its reachable set is computed. The model is wrong where
the pendulum is nonlinear, so the reachable set alone misses trajectories. Split conformal
prediction repairs that from held-out calibration trajectories: the score of a trajectory is how far
it leaves the reachable boxes at its worst time, and enlarging the boxes by the (1 - alpha) quantile
of the scores makes a new trajectory stay inside with probability at least 1 - alpha, whatever the
error of the model is, as long as trajectories are exchangeable.

Syntax:   PYTHONPATH=build python examples/python/example_linear_learn_02_conformal.py [--save FILE]
Outputs:  the learned matrix, the conformal radius, the coverage on test trajectories, and the figure
See also: example_linear_learn_01_dynamics, example_nonlinear_reach_01_vanDerPol
"""
import argparse
import math

import torch

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

timeStep = 0.1
tFinal = 3.0
X0 = cora.Zonotope(cora.Tensor([1.2, 0.0]), 0.2 * cora.eye(2))
alpha = 0.1        # the miss rate: coverage of at least 1 - alpha is guaranteed
noise = 0.01       # standard deviation of the measurement noise

# Ground Truth ------------------------------------------------------------------------------

# The damped pendulum, unknown to the learner; its states are the angle and the angular velocity.
pendulum = cora.NonlinearSys(lambda x: [x[1], -cora.sin(x[0]) - 0.2 * x[1]], 2)


def measure(count, seed):
    """Noisy trajectories from `count` random points of X0: (steps + 1, 2, count)."""
    x0 = X0.randPoint(count, cora.Rng(seed))
    truth = pendulum.simulate(x0, timeStep, tFinal)
    generator = torch.Generator().manual_seed(seed)
    return truth + noise * torch.randn(truth.shape, generator=generator, dtype=truth.dtype)


train, calibration, test = measure(30, 1), measure(200, 2), measure(1000, 3)

# Learning ----------------------------------------------------------------------------------

# Fit A so that the trajectories of x' = A x from the measured start points match the measurements.
torch.manual_seed(0)
A = torch.nn.Parameter(0.1 * torch.randn(2, 2, dtype=train.dtype))
optimizer = torch.optim.Adam([A], lr=0.02)
for step in range(400):
    optimizer.zero_grad()
    loss = ((cora.LinearSys(A).simulate(train[0], timeStep, tFinal) - train) ** 2).mean()
    loss.backward()
    optimizer.step()
print(f"learned A: {A.detach().numpy().round(2).tolist()}, fit error {loss.item():.5f}")

with torch.no_grad():
    R = cora.LinearSys(A).reach(X0, timeStep, tFinal, taylorTerms=6)
inf = torch.stack([Z.interval().inf for Z in R.timePoint])[:, :, None]  # (steps + 1, 2, 1)
sup = torch.stack([Z.interval().sup for Z in R.timePoint])[:, :, None]

# Conformal Prediction ----------------------------------------------------------------------


def scores(trajectories):
    """How far every trajectory (steps + 1, 2, N) leaves the reachable boxes at its worst time."""
    outside = torch.relu(inf - trajectories) + torch.relu(trajectories - sup)
    return outside.amax(dim=(0, 1))


# The ceil((n + 1) (1 - alpha)) / n quantile of the calibration scores.
n = calibration.shape[2]
level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
radius = torch.quantile(scores(calibration), level, interpolation="higher").item()

# Evaluation --------------------------------------------------------------------------------

covered = (scores(test) <= radius).double().mean().item()
plain = (scores(test) <= 0).double().mean().item()
print(f"conformal radius: {radius:.4f}")
print(f"coverage of the reachable set alone: {plain:.3f}")
# The guarantee is on average over calibration sets: one set of 200 varies by about +-0.02.
print(f"coverage with the conformal radius: {covered:.3f} (target {1 - alpha:.2f})")

# Visualization -----------------------------------------------------------------------------

cora.useCORAcolors("CORA:contDynamics")
fig, (phase, histogram) = plt.subplots(1, 2, figsize=(11, 4.5))

plt.sca(phase)
for i in range(40):
    phase.plot(test[:, 0, i], test[:, 1, i], color="0.3", linewidth=0.6, zorder=5,
               label="Test trajectories" if i == 0 else None)
for k in range(len(R.timePoint)):
    cora.plot(cora.Interval((inf[k, :, 0] - radius), (sup[k, :, 0] + radius)), filled=False,
              color="CORA:red", label="Enlarged by the conformal radius" if k == 0 else None)
cora.plot(R, label="Reachable set of the learned model")
cora.plot(X0, label="Initial set")
phase.set_xlabel("angle")
phase.set_ylabel("angular velocity")
phase.legend(loc="lower left", fontsize=8)

histogram.hist(scores(calibration).numpy(), bins=30, color="0.7", label="Calibration scores")
histogram.axvline(radius, color="tab:red", label=f"Quantile at {1 - alpha:.0%}: {radius:.3f}")
histogram.set_xlabel("largest distance to the reachable boxes")
histogram.legend(fontsize=8)

if args.save:
    plt.savefig(args.save, dpi=130, bbox_inches="tight")
else:
    plt.show()

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
