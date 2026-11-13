"""test_neural_network - NeuralNetwork in Python: points, the zonotope enclosure and its gradients.

CORA authors: Tobias Ladner, Lukas Koller
Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: nn/@neuralNetwork/evaluate.m
# CORA: nn/@neuralNetwork/evaluate_.m
# CORA: nn/layers/nonlinear/nnReLULayer.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotope.m
# CORA: nn/@neuralNetwork/neuralNetwork.m
# CORA: nn/@neuralNetwork/calcSensitivity.m

import unittest


try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch") from None

import cora


if not cora.has_torch:  # torch is installed, but this build has no torch backend
    raise unittest.SkipTest("needs a build with torch")

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

torch.manual_seed(0)


def layers(sizes):
    return [(torch.randn(o, i, dtype=torch.float64), 0.5 * torch.randn(o, dtype=torch.float64))
            for i, o in zip(sizes[:-1], sizes[1:], strict=True)]


def torch_forward(parameters, x):
    """The same network in plain torch: x (in, N)."""
    for k, (W, b) in enumerate(parameters):
        x = W @ x + b[:, None]
        if k + 1 < len(parameters):
            x = torch.relu(x)
    return x


# ============================================  MAIN  ============================================ #

class Evaluate(unittest.TestCase):
    def test_points_agree_with_torch(self):
        # the same network in torch, on 50 points and on one
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
        y = net.evaluate(X.rand_point(500, cora.Rng(1)))
        self.assertTrue(bool((y >= box.inf[:, None] - 1e-9).all() and (y <= box.sup[:, None] + 1e-9).all()))

    def test_the_regression_line_contains_the_outputs_and_sensitivity_is_the_jacobian(self):
        parameters = layers([2, 10, 10, 3])
        net = cora.NeuralNetwork(parameters)
        X = cora.Zonotope(torch.tensor([0.3, -0.2], dtype=torch.float64),
                          0.2 * torch.eye(2, dtype=torch.float64))
        # the regression line encloses the outputs of sampled points
        box = net.evaluate(X, poly_method="regression").interval()
        y = net.evaluate(X.rand_point(500, cora.Rng(1)))
        self.assertTrue(bool((y >= box.inf[:, None] - 1e-9).all() and (y <= box.sup[:, None] + 1e-9).all()))
        # the Jacobian at a point, against autograd
        x = torch.tensor([0.1, 0.4], dtype=torch.float64)
        reference = torch.autograd.functional.jacobian(
            lambda v: torch_forward(parameters, v[:, None])[:, 0], x)
        self.assertTrue(torch.allclose(net.sensitivity(x), reference, atol=1e-10))
        with self.assertRaises(ValueError):
            net.evaluate(X, poly_method="taylor")

    def test_the_bracket_error_contains_the_outputs(self):
        net = cora.NeuralNetwork(layers([2, 10, 10, 3]), ["tanh", "sigmoid", "identity"])
        X = cora.Zonotope(torch.tensor([0.3, -0.2], dtype=torch.float64),
                          0.2 * torch.eye(2, dtype=torch.float64))
        y = net.evaluate(X.rand_point(500, cora.Rng(1)))
        for error in ("extremum", "bracket"):
            box = net.evaluate(X, poly_method="regression", regression_error=error).interval()
            inside = (y >= box.inf[:, None] - 1e-9).all() and (y <= box.sup[:, None] + 1e-9).all()
            self.assertTrue(bool(inside))
        with self.assertRaises(ValueError):
            net.evaluate(X, poly_method="regression", regression_error="exact")
        # the polynomial zonotope keeps its factors and is enclosed by the zonotope's pairs
        P = cora.PolyZonotope(X)
        self.assertEqual(net.evaluate_with_input(P).dim, 5)

    def test_a_batch_of_sets_is_one_call(self):
        parameters = layers([2, 6, 2])
        net = cora.NeuralNetwork(parameters)
        c = torch.randn(4, 2, dtype=torch.float64)
        # a batch of sets
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
        for W, _b in parameters:
            self.assertTrue(torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0)
        self.assertTrue(torch.isfinite(c.grad).all())

    def test_wrong_arguments_are_described(self):
        W = torch.ones(2, 2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "bias"):
            cora.NeuralNetwork([(W, torch.zeros(3, dtype=torch.float64))])
        with self.assertRaisesRegex(ValueError, "inputs"):
            # layers must fit
            cora.NeuralNetwork([(W, torch.zeros(2, dtype=torch.float64)),
                                (torch.ones(1, 3, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))])
        net = cora.NeuralNetwork([(W, torch.zeros(2, dtype=torch.float64))])
        with self.assertRaises(ValueError):
            net.evaluate(torch.ones(3, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
