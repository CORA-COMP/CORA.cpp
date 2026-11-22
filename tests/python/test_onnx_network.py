"""test_onnx_network - ONNX networks on torch against onnxruntime, and their cora.NeuralNetwork

Small graphs of every operator the loader knows, built with onnx.helper: the batched torch run
equals onnxruntime's (float32) to float32 accuracy; a chain of affine maps and activations lowers
to dense layers whose network equals the graph in float64, and a plain Gemm keeps its weights bit
for bit; a residual connection or a max pool is no chain and says why.

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import unittest


try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # the VNN-COMP front end needs onnx, onnxruntime and torch
    raise unittest.SkipTest("needs onnx, onnxruntime and torch") from None

import cora


if not cora.has_torch:
    raise unittest.SkipTest("needs a build with torch") from None

from cora.onnx_network import OnnxNetwork


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class TorchRun(unittest.TestCase):
    def test_every_operator_agrees_with_onnxruntime(self):
        # each graph on 7 rows at once, against onnxruntime one sample at a time
        for m in (mlp(), conv_net(), residual_net(), maxpool_net()):
            net = OnnxNetwork(m, dtype=torch.float32)
            x = RNG.uniform(-1, 1, (7, net.num_inputs)).astype(np.float32)
            expected = onnxruntime_rows(m, x)
            np.testing.assert_allclose(net(torch.from_numpy(x)).numpy(), expected, rtol=1e-5, atol=1e-5)
            # the same graph in float64 is as close
            y64 = net.cast(torch.float64)(torch.from_numpy(x).double()).numpy()
            np.testing.assert_allclose(y64, expected, rtol=1e-5, atol=1e-5)

    def test_gradients_flow(self):
        # autograd through the batched run, as PGD uses it
        net = OnnxNetwork(conv_net(), dtype=torch.float32)
        x = torch.zeros(3, net.num_inputs, requires_grad=True)
        (grad,) = torch.autograd.grad(net(x).sum(), x)
        self.assertEqual(tuple(grad.shape), (3, 16))
        self.assertTrue(bool(torch.isfinite(grad).all()))


class Lowering(unittest.TestCase):
    def test_chains_lower_to_the_same_function(self):
        # the dense layers in float64 equal the graph in float64
        for m in (mlp(), conv_net()):
            net = OnnxNetwork(m, dtype=torch.float64)
            layers, why = net.dense_layers()
            self.assertIsNone(why)
            x = RNG.uniform(-1, 1, (9, net.num_inputs))
            np.testing.assert_allclose(
                dense_rows(layers, x), net(torch.from_numpy(x)).numpy(), rtol=1e-12, atol=1e-12
            )

    def test_activations_and_weights(self):
        # the chain of the MLP: relu, sigmoid, tanh, identity; the Gemm's weights exactly
        m = mlp()
        layers, _ = OnnxNetwork(m).dense_layers()
        self.assertEqual(layers[1], ["relu", "sigmoid", "tanh", "identity"])
        W2 = numpy_helper.to_array(m.graph.initializer[3])
        self.assertTrue(np.array_equal(layers[0][1][0].numpy(), W2.astype(np.float64)))

    def test_neural_network_evaluates_like_the_graph(self):
        # the cora.NeuralNetwork of the chain at points
        net = OnnxNetwork(mlp(), dtype=torch.float64)
        nn, why = net.to_neural_network()
        self.assertIsNone(why)
        x = torch.as_tensor(RNG.uniform(-1, 1, (5, 5)))
        np.testing.assert_allclose(np.asarray(nn.evaluate(x)), net(x).numpy(), rtol=1e-12, atol=1e-12)

    def test_what_is_no_chain(self):
        # a residual connection and a max pool, each with the reason
        for m, reason in ((residual_net(), "residual"), (maxpool_net(), "MaxPool")):
            nn, why = OnnxNetwork(m).to_neural_network()
            self.assertIsNone(nn)
            self.assertIn(reason, why)

    def test_convolutions_lower_layer_by_layer(self):
        # with a Conv the layers are conv2d and gather dicts, never a matrix of the convolution
        net = OnnxNetwork(conv_net(), dtype=torch.float64)
        lowered, why = net.layers()
        self.assertIsNone(why)
        kinds = [next(iter(layer)) if isinstance(layer, dict) else "dense" for layer in lowered[0]]
        self.assertIn("conv2d", kinds)
        self.assertIn("gather", kinds)
        x = RNG.uniform(-1, 1, (9, net.num_inputs))
        np.testing.assert_allclose(
            structured_rows(*lowered, x), net(torch.from_numpy(x)).numpy(), rtol=1e-12, atol=1e-12
        )

    def test_size_cap(self):
        # max_elements refuses unit-vector batches beyond it
        _, why = OnnxNetwork(conv_net()).dense_layers(max_elements=100)
        self.assertIn("max_elements", why)


class Loading(unittest.TestCase):
    def test_shapes_and_operators(self):
        net = OnnxNetwork(conv_net())
        self.assertEqual((net.input_shape, net.num_inputs, net.num_outputs), ([1, 1, 4, 4], 16, 4))
        self.assertIn("Conv", net.op_types())
        self.assertGreater(net.activation_elements(), 16)

    def test_unknown_operator(self):
        m = model([helper.make_node("Erf", ["x"], ["y"])], [("x", [1, 2])], [("y", [1, 2])], {})
        with self.assertRaisesRegex(NotImplementedError, "Erf"):
            OnnxNetwork(m)(torch.zeros(1, 2))


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

RNG = np.random.default_rng(3)


def weight(*shape):
    """A random float32 array."""
    return RNG.standard_normal(shape).astype(np.float32)


def model(nodes, inputs, outputs, initializers):
    """A model of the nodes with float32 inputs and outputs (name, shape)."""
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        [numpy_helper.from_array(a, name) for name, a in initializers.items()],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def onnxruntime_rows(m, x):
    """onnxruntime's outputs of the rows x, one sample of the declared shape at a time."""
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    return np.stack([sess.run(None, {inp.name: xi.reshape(shape)})[0].reshape(-1) for xi in x])


def dense_rows(layers, x):
    """The lowered layers on the rows x in float64."""
    h = torch.as_tensor(x, dtype=torch.float64)
    act = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "identity": lambda t: t}
    for (W, b), a in zip(*layers, strict=True):
        h = act[a](h @ W.T + b)
    return h.numpy()


def mlp():
    """Sub, Flatten, MatMul + Add, Relu, Gemm, Sigmoid, Mul, Div, Tanh: acasxu-like."""
    nodes = [
        helper.make_node("Sub", ["x", "mean"], ["s"]),
        helper.make_node("Flatten", ["s"], ["f"]),
        helper.make_node("MatMul", ["f", "W1"], ["m"]),
        helper.make_node("Add", ["m", "b1"], ["a"]),
        helper.make_node("Relu", ["a"], ["r"]),
        helper.make_node("Gemm", ["r", "W2", "b2"], ["g"], transB=1),
        helper.make_node("Sigmoid", ["g"], ["sg"]),
        helper.make_node("Mul", ["sg", "k"], ["mu"]),
        helper.make_node("Div", ["mu", "k2"], ["d"]),
        helper.make_node("Tanh", ["d"], ["t"]),
        helper.make_node("Gemm", ["t", "W3", "b3"], ["y"]),
    ]
    # the constants of the graph
    init = {
        "mean": weight(1, 1, 1, 5),
        "W1": weight(5, 8),
        "b1": weight(8),
        "W2": weight(6, 8),
        "b2": weight(6),
        "k": weight(6),
        "k2": np.full(6, 2.0, np.float32),
        "W3": weight(6, 3),
        "b3": weight(3),
    }
    return model(nodes, [("x", [1, 1, 1, 5])], [("y", [1, 3])], init)


def structured_rows(layers, activations, x):
    """The layers of OnnxNetwork.layers on the rows x, by torch (the reference)."""
    import torch.nn.functional as F

    h = torch.as_tensor(x, dtype=torch.float64)
    acts = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}
    for layer, act in zip(layers, activations, strict=True):
        # a dense layer, a gather, a ConvTranspose or a convolution
        if not isinstance(layer, dict):
            h = h @ layer[0].T + layer[1]
        elif "gather" in layer:
            padded = torch.cat([h, torch.zeros(h.shape[0], 1, dtype=h.dtype)], 1)
            index = [i if i >= 0 else h.shape[1] for i in layer["gather"]]
            h = padded[:, index] + layer["bias"]
        elif "conv_transpose2d" in layer:
            c, hh, ww = layer["output"]
            # the transpose of the convolution on the output: by autograd of the convolution
            t, left, b, r = layer["pads"]
            z0 = torch.zeros(h.shape[0], c, hh, ww, dtype=h.dtype, requires_grad=True)
            z = F.conv2d(
                F.pad(z0, (left, r, t, b)),
                layer["conv_transpose2d"],
                stride=layer["stride"],
                dilation=layer["dilation"],
                groups=layer["groups"],
            )
            (g,) = torch.autograd.grad(z, z0, h.reshape(z.shape))
            h = g.reshape(h.shape[0], -1) + layer["bias"]
        else:
            # a convolution: the pads first, as ONNX's may be asymmetric
            c, hh, ww = layer["input"]
            t, left, b, r = layer["pads"]
            z = F.pad(h.reshape(-1, c, hh, ww), (left, r, t, b))
            z = F.conv2d(
                z, layer["conv2d"], stride=layer["stride"], dilation=layer["dilation"], groups=layer["groups"]
            )
            h = z.reshape(h.shape[0], -1) + layer["bias"]
        h = acts[act](h) if act in acts else h
    return h.numpy()


def conv_net():
    """Conv (asymmetric pads), BatchNormalization, Relu, AveragePool, Transpose, Reshape (to a
    constant shape of batch 1), Concat with a constant, Dropout, Identity, MatMul."""
    nodes = [
        helper.make_node("Conv", ["x", "Wc", "bc"], ["c"], pads=[1, 0, 0, 1], strides=[1, 1]),
        helper.make_node("BatchNormalization", ["c", "s", "o", "mu", "var"], ["bn"]),
        helper.make_node("Relu", ["bn"], ["r"]),
        helper.make_node("AveragePool", ["r"], ["p"], kernel_shape=[2, 2], strides=[1, 1]),
        helper.make_node("Transpose", ["p"], ["t"], perm=[0, 1, 3, 2]),
        helper.make_node("Reshape", ["t", "shape"], ["f"]),
        helper.make_node("Concat", ["f", "extra"], ["cat"], axis=1),
        helper.make_node("Dropout", ["cat"], ["dr"]),
        helper.make_node("Identity", ["dr"], ["id"]),
        helper.make_node("MatMul", ["id", "W"], ["y"]),
    ]
    # the constants of the graph
    init = {
        "Wc": weight(2, 1, 2, 2),
        "bc": weight(2),
        "s": weight(2),
        "o": weight(2),
        "mu": weight(2),
        "var": np.abs(weight(2)) + 0.5,
        "shape": np.array([1, 18], np.int64),
        "extra": weight(1, 2),
        "W": weight(20, 4),
    }
    return model(nodes, [("x", [1, 1, 4, 4])], [("y", [1, 4])], init)


def residual_net():
    """A residual connection: y = W2 relu(W1 x) + x."""
    nodes = [
        helper.make_node("MatMul", ["x", "W1"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("MatMul", ["r", "W2"], ["m"]),
        helper.make_node("Add", ["m", "x"], ["y"]),
    ]
    return model(nodes, [("x", [1, 3])], [("y", [1, 3])], {"W1": weight(3, 4), "W2": weight(4, 3)})


def maxpool_net():
    """A max pool (ONNX's default stride 1), which is no affine map."""
    nodes = [
        helper.make_node("MaxPool", ["x"], ["p"], kernel_shape=[2, 2]),
        helper.make_node("Flatten", ["p"], ["y"]),
    ]
    return model(nodes, [("x", [1, 1, 4, 4])], [("y", [1, 9])], {})


# ----------------------------------------  END OF CODE  ----------------------------------------- #
