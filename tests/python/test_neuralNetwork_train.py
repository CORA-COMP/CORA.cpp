"""test_neuralNetwork_train - a loss over the output set of a network updates its weights"""
import unittest

import torch
from torch import nn

import cora

dtype = torch.float64


def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)).to(dtype)


def output_set(m, x, eps):
    """The output zonotopes of the eps-boxes around the points x, from the current weights."""
    net = cora.NeuralNetwork([(l.weight, l.bias) for l in m if isinstance(l, nn.Linear)])
    return net.evaluate(cora.Zonotope(x, eps * torch.eye(2, dtype=dtype).expand(len(x), 2, 2)))


def f_radius(Y):
    return ((Y.G ** 2).sum(dim=(-2, -1)) + 1e-12).sqrt() / Y.G.shape[-2]


def set_loss(m, x, labels, eps, tau):
    """Set-based loss: (1 - tau) CE(center of Y, t) + tau / eps * F-radius of Y."""
    Y = output_set(m, x, eps)
    return (1 - tau) * nn.functional.cross_entropy(Y.c, labels) + tau / eps * f_radius(Y).mean()


x = torch.tensor([[0.1, 0.7], [0.6, 0.9], [0.5, 0.5], [0.9, 0.1]], dtype=dtype)
labels = torch.tensor([1, 1, 0, 0])


class SetBasedTraining(unittest.TestCase):
    def test_one_step_on_the_set_loss_updates_every_weight(self):
        m = model()
        before = [p.detach().clone() for p in m.parameters()]
        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss = set_loss(m, x, labels, eps=0.05, tau=0.1)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        for old, p in zip(before, m.parameters()):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertFalse(torch.equal(old, p.detach()))

    def test_minimizing_the_set_size_shrinks_the_output_sets(self):
        m = model()
        optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
        sizes = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = set_loss(m, x, labels, eps=0.05, tau=1.0)  # only the size of the output sets
            sizes.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(sizes[-1], sizes[0])


if __name__ == "__main__":
    unittest.main()
