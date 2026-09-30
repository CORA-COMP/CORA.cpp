"""test_linear_learn_conformal - a conformal quantile enlarges the reachable boxes to cover trajectories"""
import math
import unittest

try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch")

import cora

X0 = cora.Zonotope(cora.Tensor([1.2, 0.0]), 0.2 * cora.eye(2))
PENDULUM = cora.NonlinearSys(lambda x: [x[1], -cora.sin(x[0]) - 0.2 * x[1]], 2)
MODEL = cora.LinearSys(cora.Tensor([[0.0, 1.0], [-1.0, -0.2]]))  # the linearization at the origin


def trajectories(count, seed):
    return PENDULUM.simulate(X0.randPoint(count, cora.Rng(seed)), 0.1, 2.0)


def scores(trajectories, inf, sup):
    """How far every trajectory (steps + 1, 2, N) leaves the boxes at its worst time."""
    outside = torch.relu(inf - trajectories) + torch.relu(trajectories - sup)
    return outside.amax(dim=(0, 1))


class Conformal(unittest.TestCase):
    def test_the_quantile_of_the_scores_gives_the_promised_coverage(self):
        R = MODEL.reach(X0, 0.1, 2.0, taylorTerms=6)
        inf = torch.stack([Z.interval().inf for Z in R.timePoint])[:, :, None]
        sup = torch.stack([Z.interval().sup for Z in R.timePoint])[:, :, None]
        calibration = scores(trajectories(100, 1), inf, sup)
        test = scores(trajectories(500, 2), inf, sup)

        alpha = 0.2
        level = math.ceil((len(calibration) + 1) * (1 - alpha)) / len(calibration)
        radius = torch.quantile(calibration, level, interpolation="higher")
        # the linear model is wrong for the pendulum: the boxes alone miss trajectories
        self.assertLess((test <= 0).double().mean(), 0.7)
        # one calibration set varies by about 0.04 around 1 - alpha
        self.assertGreater((test <= radius).double().mean(), 1 - alpha - 0.1)


if __name__ == "__main__":
    unittest.main()
