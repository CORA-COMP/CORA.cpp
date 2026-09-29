"""test_linear_learn - the system matrix of a reach computation is a parameter that gradient steps update"""
import unittest

import torch

import cora

X0 = cora.Zonotope(cora.Tensor([1.0, 0.0]), 0.1 * cora.eye(2))
A_TRUE = cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]])


def loss(A, measurements):
    """How far the measurements (steps + 1, 2, N) leave the boxes around the time-point sets."""
    R = cora.LinearSys(A).reach(X0, 0.1, 1.0, taylorTerms=6)
    boxes = [Z.interval() for Z in R.timePoint]
    inf = torch.stack([b.inf for b in boxes])[:, :, None]
    sup = torch.stack([b.sup for b in boxes])[:, :, None]
    return (torch.relu(inf - measurements) + torch.relu(measurements - sup)).mean()


class LearnDynamics(unittest.TestCase):
    def setUp(self):
        self.measurements = cora.LinearSys(A_TRUE).simulate(X0.randPoint(10, cora.Rng(0)), 0.1, 1.0)

    def test_a_gradient_step_updates_the_matrix(self):
        A = torch.nn.Parameter(cora.Tensor([[0.0, 0.5], [-0.5, 0.0]]))
        before = A.detach().clone()
        value = loss(A, self.measurements)
        value.backward()
        torch.optim.Adam([A], lr=0.02).step()
        self.assertTrue(torch.isfinite(A.grad).all() and A.grad.abs().sum() > 0)
        self.assertFalse(torch.equal(before, A.detach()))

    def test_a_few_steps_bring_the_measurements_inside(self):
        A = torch.nn.Parameter(cora.Tensor([[0.0, 0.5], [-0.5, 0.0]]))
        optimizer = torch.optim.Adam([A], lr=0.02)
        first = loss(A, self.measurements).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss(A, self.measurements).backward()
            optimizer.step()
        self.assertLess(loss(A, self.measurements).item(), first)


if __name__ == "__main__":
    unittest.main()
