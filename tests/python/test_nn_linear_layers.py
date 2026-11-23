"""test_nn_linear_layers - CORA's linear layers as layers of a NeuralNetwork, in Python.

CORA authors: Tobias Ladner, Lukas Koller
Authors:       Claude Sonnet 5.5
Written:       23-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: nn/layers/linear/nnLinearLayer.m
# CORA: nn/layers/linear/nnIdentityLayer.m
# CORA: nn/layers/linear/nnElementwiseAffineLayer.m
# CORA: nn/layers/linear/nnBatchNormLayer.m
# CORA: nn/layers/linear/nnAvgPool2DLayer.m
# CORA: nn/layers/linear/nnLipConstrLinearLayer.m
# CORA: nn/@neuralNetwork/getNumNeurons.m
# CORA: nn/@neuralNetwork/getNumParams.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def run(layer, x):
    """The network of one layer without activation on the point x."""
    net = cora.NeuralNetwork([layer], ["identity"])
    return np.asarray(net.evaluate(np.asarray(x, dtype=float)))


# the layers on points, by the definitions of CORA's layers
class LinearLayers(unittest.TestCase):
    def test_identity_and_elementwise_affine(self):
        np.testing.assert_allclose(run({"identity": 3}, [1.0, -2.0, 3.0]), [1.0, -2.0, 3.0])
        layer = {"elementwise_affine": [2.0, -1.0, 0.5], "offset": [1.0], "inputs": 3}
        np.testing.assert_allclose(run(layer, [1.0, -2.0, 3.0]), [3.0, 3.0, 2.5])

    def test_batch_norm_inference(self):
        # two channels of two pixels, the folded map against the three steps
        eps = 1e-5
        layer = {"batch_norm": [2.0, 1.0], "offset": [0.5, -1.0], "mean": [1.0, 0.0],
                 "var": [3.0, 1.0], "input": [2, 1, 2], "epsilon": eps}
        f0, f1 = 2.0 / math.sqrt(3.0 + eps), 1.0 / math.sqrt(1.0 + eps)
        want = [0.5, 3.0 * f0 + 0.5, 2.0 * f1 - 1.0, -2.0 * f1 - 1.0]
        np.testing.assert_allclose(run(layer, [1.0, 4.0, 2.0, -2.0]), want, atol=1e-12)

    def test_average_pooling(self):
        layer = {"avg_pool2d": (2, 2), "input": [1, 2, 4]}
        np.testing.assert_allclose(run(layer, [1, 2, 5, 6, 3, 4, 7, 8]), [2.5, 6.5])

    def test_lipschitz_constrained_layer_and_counts(self):
        # row 0 has the absolute sum 4 above lambda 2: halved
        layer = {"lip_constr_linear": [[3.0, -1.0], [0.25, 0.25]], "bias": [4.0, 1.0],
                 "lambda": 2.0}
        np.testing.assert_allclose(run(layer, [1.0, 1.0]), [3.0, 1.5])
        net = cora.NeuralNetwork([([[3.0, -1.0], [0.25, 0.25]], [4.0, 1.0]),
                                  ([[1.0, 2.0]], [0.0])])
        self.assertEqual(list(net.num_neurons()), [2, 2, 1])
        self.assertEqual(net.num_params(), 9)

    def test_wrong_input_is_refused(self):
        with self.assertRaisesRegex(Exception, "pool size"):
            run({"avg_pool2d": (0, 2), "input": [1, 2, 4]}, [0.0] * 8)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
