"""test_neuralNetwork - NeuralNetwork in Python: points, the zonotope enclosure and its gradients."""
import unittest

import torch

import cora

torch.manual_seed(0)


def layers(sizes):
    return [(torch.randn(o, i, dtype=torch.float64), 0.5 * torch.randn(o, dtype=torch.float64))
            for i, o in zip(sizes[:-1], sizes[1:])]


def torch_forward(parameters, x):
    """The same network in plain torch: x (in, N)."""
    for k, (W, b) in enumerate(parameters):
        x = W @ x + b[:, None]
        if k + 1 < len(parameters):
            x = torch.relu(x)
    return x


class Evaluate(unittest.TestCase):
    def test_points_agree_with_torch(self):
        parameters = layers([3, 8, 8, 2])
        x = torch.randn(3, 50, dtype=torch.float64)
        y = cora.NeuralNetwork(parameters).evaluate(x)
        self.assertTrue(torch.allclose(y, torch_forward(parameters, x), atol=1e-12))
        one = cora.NeuralNetwork(parameters).evaluate(x[:, 0])
        self.assertEqual(one.shape, (2,))

    def test_the_output_set_contains_the_outputs(self):
        parameters = layers([2, 10, 10, 3])
        net = cora.NeuralNetwork(parameters)
        X = cora.Zonotope(torch.tensor([0.3, -0.2], dtype=torch.float64),
                          0.2 * torch.eye(2, dtype=torch.float64))
        box = net.evaluate(X).interval()
        y = net.evaluate(X.randPoint(500, cora.Rng(1)))
        self.assertTrue(bool((y >= box.inf[:, None] - 1e-9).all() and (y <= box.sup[:, None] + 1e-9).all()))

    def test_a_batch_of_sets_is_one_call(self):
        parameters = layers([2, 6, 2])
        net = cora.NeuralNetwork(parameters)
        c = torch.randn(4, 2, dtype=torch.float64)
        G = 0.1 * torch.eye(2, dtype=torch.float64).expand(4, 2, 2)
        batch = net.evaluate(cora.Zonotope(c, G)).interval()
        for i in range(4):
            single = net.evaluate(cora.Zonotope(c[i], G[i])).interval()
            self.assertTrue(torch.allclose(batch.inf[i], single.inf, atol=1e-12))
            self.assertTrue(torch.allclose(batch.sup[i], single.sup, atol=1e-12))

    def test_gradients_reach_the_weights_and_the_set(self):
        parameters = [(W.clone().requires_grad_(), b.clone().requires_grad_()) for W, b in layers([2, 6, 2])]
        c = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
        Y = cora.NeuralNetwork(parameters).evaluate(cora.Zonotope(c, 0.1 * torch.eye(2, dtype=torch.float64)))
        box = Y.interval()
        (box.sup - box.inf).sum().backward()
        for W, b in parameters:
            self.assertTrue(torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0)
        self.assertTrue(torch.isfinite(c.grad).all())

    def test_wrong_arguments_are_described(self):
        W = torch.ones(2, 2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "bias"):
            cora.NeuralNetwork([(W, torch.zeros(3, dtype=torch.float64))])
        with self.assertRaisesRegex(ValueError, "inputs"):
            cora.NeuralNetwork([(W, torch.zeros(2, dtype=torch.float64)),
                                (torch.ones(1, 3, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))])
        net = cora.NeuralNetwork([(W, torch.zeros(2, dtype=torch.float64))])
        with self.assertRaises(ValueError):
            net.evaluate(torch.ones(3, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
