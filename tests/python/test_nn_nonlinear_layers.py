"""test_nn_nonlinear_layers - CORA's exp, softmax, max-pool layers, batches and gradients, in Python.

CORA authors: Tobias Ladner, Lukas Koller
Authors:       Claude Sonnet 5.5
Written:       23-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: nn/layers/nonlinear/nnExpLayer.m
# CORA: nn/layers/nonlinear/nnSoftmaxLayer.m
# CORA: nn/layers/nonlinear/nnMaxPool2DLayer.m
# CORA: nn/@neuralNetwork/evaluateZonotopeBatch.m
# CORA: nn/@neuralNetwork/backprop.m
# CORA: nn/@neuralNetwork/backpropIntervalBatch.m
# CORA: nn/@neuralNetwork/propagateBounds.m
# CORA: nn/@neuralNetwork/getInputNeuronOrder.m
# CORA: nn/@neuralNetwork/getNormalForm.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def network():
    """Two inputs, a sigmoid layer of three, a ReLU layer of two, an output."""
    return cora.NeuralNetwork(
        [([[1.0, -0.5], [0.3, 0.8], [-0.7, 0.4]], [0.2, -0.1, 0.0]),
         ([[0.5, 1.0, -1.0], [1.0, -0.2, 0.3]], [0.1, 0.0]),
         ([[1.0, 2.0]], [0.5])],
        ["sigmoid", "relu", "identity"])


# the layers on points, by the definitions of CORA's layers
class NonlinearLayers(unittest.TestCase):
    def test_exp_softmax_and_max_pool(self):
        x = np.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(cora.NeuralNetwork.exp_layer(3).evaluate(x), np.exp(x))
        # CORA's softmax shifts by the maximum
        e = np.exp(x - x.max())
        np.testing.assert_allclose(cora.NeuralNetwork.softmax_layer(3).evaluate(x), e / e.sum())
        # a 4 x 4 image of two by two windows
        img = np.arange(16.0).reshape(1, 4, 4)[:, :, ::-1].copy()
        pool = cora.NeuralNetwork.max_pool2d_layer((2, 2), [1, 4, 4])
        want = img.reshape(1, 2, 2, 2, 2).max(axis=(2, 4)).reshape(-1)
        np.testing.assert_allclose(pool.evaluate(img.reshape(-1)), want)

    def test_zonotope_batch(self):
        # the box around a point, a generator per input
        nn = network()
        c, G = np.array([[0.4, 0.9]]), 0.1 * np.eye(2)[None]
        out_c, out_G, dr = nn.evaluate_zonotope_batch(c, G)
        self.assertIsNone(dr)
        # a point of the box lies in the enclosure
        y = np.asarray(nn.evaluate(np.array([0.45, 0.85])))
        radius = np.abs(out_G).sum(axis=1)[0]
        self.assertTrue(np.all(np.abs(y - out_c[0]) <= radius + 1e-12))
        # one error generator per layer leaves the rest to the radius
        _, _, dr = nn.evaluate_zonotope_batch(c, G, num_approx_err=1)
        self.assertIsNotNone(dr)

    def test_gradients_and_bounds(self):
        # the gradient against the Jacobian, then the bounds on a point box
        nn = network()
        x, g = np.array([[0.4, 0.9]]), np.array([[2.0]])
        got = np.asarray(nn.backprop(x, g))
        np.testing.assert_allclose(got[0], 2.0 * np.asarray(nn.sensitivity(x[0]))[0], atol=1e-12)
        # a point box gives the same gradient on both sides
        lo, hi = nn.backprop_interval_batch(x, x, g, g)
        np.testing.assert_allclose(lo, got, atol=1e-12)
        np.testing.assert_allclose(hi, got, atol=1e-12)
        # the bounds reach the next activation layer only
        bounds = nn.propagate_bounds([([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]), None, None], 0)
        self.assertIsNotNone(bounds[1])
        self.assertIsNone(bounds[2])

    def test_order_and_normal_form(self):
        nn = cora.NeuralNetwork([([[3.0, 0.0, 1.0, -2.0, 0.0, 0.5]], [0.0])], ["identity"])
        x = np.zeros(6)
        self.assertEqual(list(nn.input_neuron_order("snake", x, [1, 2, 3])), [0, 1, 2, 5, 4, 3])
        self.assertEqual(list(nn.input_neuron_order("sensitivity", x)), [1, 4, 5, 2, 3, 0])
        deep = cora.NeuralNetwork([([[1.0, 2.0]], [0.1]), ([[2.0]], [0.0])], ["identity", "relu"])
        normal = deep.normal_form()
        np.testing.assert_allclose(normal.evaluate(np.array([0.3, 0.4])),
                                   deep.evaluate(np.array([0.3, 0.4])))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
