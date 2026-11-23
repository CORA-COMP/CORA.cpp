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
# CORA: nn/layers/nonlinear/nnSoftPlusLayer.m
# CORA: nn/layers/nonlinear/nnRootLayer.m
# CORA: nn/layers/nonlinear/nnInvSqrtRootLayer.m
# CORA: nn/layers/nonlinear/nnGroupSortLayer.m
# CORA: nn/layers/other/nnCompositeLayer.m
# CORA: nn/layers/linear/nnGeneratorReductionLayer.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/getMergeBuckets.m
# CORA: nn/@neuralNetwork/computeReducedNetwork.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/findRegionPolys.m
# CORA: nn/layers/nonlinear/nnSquareLayer.m
# CORA: nn/layers/linear/nnConv2DLayer.m

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

    def test_batch_three_layers(self):
        # softplus of sharpness 2, root, inverse square root, and a group sort on points
        x = np.array([0.5, 2.0, 1.0])
        np.testing.assert_allclose(cora.NeuralNetwork.softplus_layer(3, 2.0).evaluate(x),
                                   np.log1p(np.exp(2 * x)) / 2)
        np.testing.assert_allclose(cora.NeuralNetwork.root_layer(3).evaluate(x), np.sqrt(x))
        np.testing.assert_allclose(cora.NeuralNetwork.inv_sqrt_root_layer(3).evaluate(x), x ** -0.5)
        np.testing.assert_allclose(cora.NeuralNetwork.group_sort_layer(3, 2).evaluate(x),
                                   [0.5, 2.0, 1.0])
        np.testing.assert_allclose(cora.NeuralNetwork.group_sort_layer(3, 3).evaluate(x),
                                   [0.5, 1.0, 2.0])
        # a composite of two branches: their sum, their concatenation, a skip connection
        branch = cora.NeuralNetwork.exp_layer(3)
        total = cora.NeuralNetwork.composite_layer([branch, None], "add", 3)
        np.testing.assert_allclose(total.evaluate(x), np.exp(x) + x)
        joined = cora.NeuralNetwork.composite_layer([branch, None], "concat", 3)
        np.testing.assert_allclose(joined.evaluate(x), np.concatenate([np.exp(x), x]))
        with self.assertRaises(ValueError):
            cora.NeuralNetwork.composite_layer([branch], "mean", 3)

    def test_square_and_conv(self):
        # x^2 on points, and a 1 x 1 convolution (channels 1 to 2) of a 1 x 2 x 2 image
        x = np.array([0.5, -2.0])
        np.testing.assert_allclose(
            cora.NeuralNetwork([([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])], ["square"]).evaluate(x),
            x ** 2)
        kernel = np.array([2.0, -1.0]).reshape(2, 1, 1, 1)
        conv = cora.NeuralNetwork([{"conv2d": kernel, "bias": np.zeros(8), "input": [1, 2, 2]}],
                                  ["identity"])
        img = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(conv.evaluate(img), np.concatenate([2 * img, -img]))

    def test_reduction(self):
        # the generator reduction keeps the shortest and boxes the rest
        c = np.array([[1.0, -1.0]])
        G = np.array([[[5.0, 0.0], [0.0, 1.0], [1.0, 2.0], [0.0, -2.0], [-3.0, 1.0]]])
        rc, rG = cora.NeuralNetwork.reduce_generators(c, G, 4)
        np.testing.assert_allclose(rc, c)
        np.testing.assert_allclose(np.asarray(rG)[0], [[0.0, 1.0], [0.0, -2.0], [9.0, 0.0], [0.0, 3.0]])
        self.assertEqual(cora.NeuralNetwork.merge_buckets("sigmoid"), [0.0, 1.0])
        # the regions of the sigmoid follow each other, each within the tolerance
        regions = cora.NeuralNetwork.find_region_polys(
            "sigmoid", 0.05, 2, -6.0, 6.0, [0.0], 0.0025, [1.0], 0.0025)
        self.assertEqual(regions[0]["l"], -np.inf)
        self.assertEqual(regions[-1]["u"], np.inf)
        for a, b in zip(regions, regions[1:], strict=False):
            self.assertEqual(a["u"], b["l"])
        self.assertTrue(all(r["d"] <= 0.05 + 1e-12 for r in regions))
        # a network with neurons that settle near 0 loses them, and the output set stays sound
        W = [[0.001, 0.0]] * 10
        nn = cora.NeuralNetwork(
            [(W, [0.1 * i for i in range(10)]), ([[1.0 + 0.1 * i for i in range(10)]], [0.5])],
            ["relu", "identity"])
        X = cora.Zonotope(np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.5]]))
        R = nn.compute_reduced_network(X, bucket_tol=[0.15])
        self.assertEqual(int(np.sum(R["merged"][0])), 2)
        self.assertIsNone(R["merged"][1])
        self.assertEqual(R["network"].num_neurons(), [2, 8, 1])
        self.assertIsNotNone(R["offset_center"][1])
        y = np.asarray(nn.evaluate(np.array([0.7, -0.2])))
        self.assertTrue(np.all(R["output"].contains(y)))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
