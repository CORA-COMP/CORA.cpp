"""test_lean_nn_verify - the set propagation of the VNN-COMP verifier against the CORALean oracle

Every layer type that the ONNX front end produces (a dense linear layer from any affine operator,
ReLU, sigmoid, tanh) on the lean backend at dtype nearest, bit for bit against the oracle's
nnLayer.evaluateZonotope through cora.lean.evaluate_network: the lowered networks of graphs of
Gemm, MatMul, Sub, Mul, Div, Conv, BatchNormalization, AveragePool, Transpose, Reshape, Concat,
Dropout and Identity. Then three small end-to-end runs of the C++ verify on lean (an ACAS
Xu-sized ReLU network, 5 inputs, two hidden layers): verified, falsified and undecided; the first
set of its bounding goes through the oracle (the C++ test test_lean_nn_verify replays more).

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""
import contextlib
import os
import tempfile
import unittest
import warnings


try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401
    import torch
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    raise unittest.SkipTest("needs onnx, onnxruntime and torch") from None

import cora


if not cora.has_torch:
    raise unittest.SkipTest("needs a build with torch") from None

from cora import nn_verify
from cora.onnx_network import OnnxNetwork


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

# Skips without CORACPP_ORACLE; with CORACPP_REQUIRE_ORACLE (CI) the tests run and fail instead.
needs_oracle = unittest.skipUnless(
    os.environ.get("CORACPP_ORACLE") or os.environ.get("CORACPP_REQUIRE_ORACLE", "0") not in ("", "0"),
    "needs CORACPP_ORACLE")
RNG = np.random.default_rng(5)


def weight(*shape):
    """A random float32 array of small entries."""
    return (0.5 * RNG.standard_normal(shape)).astype(np.float32)


def model(nodes, inputs, outputs, initializers):
    """A model of the nodes with float32 inputs and outputs (name, shape)."""
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        [numpy_helper.from_array(a, name) for name, a in initializers.items()])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


def dense_graph():
    """Sub, Flatten, MatMul + Add, Relu, Gemm, Sigmoid, Mul, Div, Tanh, Gemm."""
    nodes = [helper.make_node("Sub", ["x", "mean"], ["s"]), helper.make_node("Flatten", ["s"], ["f"]),
             helper.make_node("MatMul", ["f", "W1"], ["m"]), helper.make_node("Add", ["m", "b1"], ["a"]),
             helper.make_node("Relu", ["a"], ["r"]),
             helper.make_node("Gemm", ["r", "W2", "b2"], ["g"], transB=1),
             helper.make_node("Sigmoid", ["g"], ["sg"]), helper.make_node("Mul", ["sg", "k"], ["mu"]),
             helper.make_node("Div", ["mu", "k2"], ["d"]), helper.make_node("Tanh", ["d"], ["t"]),
             helper.make_node("Gemm", ["t", "W3", "b3"], ["y"])]
    # the constants of the graph
    init = {"mean": weight(1, 1, 1, 3), "W1": weight(3, 4), "b1": weight(4), "W2": weight(3, 4),
            "b2": weight(3), "k": weight(3), "k2": np.full(3, 2.0, np.float32), "W3": weight(3, 2),
            "b3": weight(2)}
    return model(nodes, [("x", [1, 1, 1, 3])], [("y", [1, 2])], init)


def conv_graph():
    """Conv, BatchNormalization, Relu, AveragePool, Transpose, Reshape, Concat, Dropout, Identity,
    MatMul."""
    nodes = [helper.make_node("Conv", ["x", "Wc", "bc"], ["c"], pads=[1, 0, 0, 1]),
             helper.make_node("BatchNormalization", ["c", "s", "o", "mu", "var"], ["bn"]),
             helper.make_node("Relu", ["bn"], ["r"]),
             helper.make_node("AveragePool", ["r"], ["p"], kernel_shape=[2, 2]),
             helper.make_node("Transpose", ["p"], ["t"], perm=[0, 1, 3, 2]),
             helper.make_node("Reshape", ["t", "shape"], ["f"]),
             helper.make_node("Concat", ["f", "extra"], ["cat"], axis=1),
             helper.make_node("Dropout", ["cat"], ["dr"]), helper.make_node("Identity", ["dr"], ["id"]),
             helper.make_node("MatMul", ["id", "W"], ["y"])]
    # the constants of the graph
    init = {"Wc": weight(1, 1, 2, 2), "bc": weight(1), "s": weight(1), "o": weight(1), "mu": weight(1),
            "var": np.abs(weight(1)) + 0.5, "shape": np.array([1, 4], np.int64), "extra": weight(1, 1),
            "W": weight(5, 2)}
    return model(nodes, [("x", [1, 1, 3, 3])], [("y", [1, 2])], init)


def lean_layers(layers):
    """The lowered layers as cora.lean.evaluate_network's list."""
    out = []
    for (W, b), act in zip(*layers, strict=True):
        out.append(("linear", W.numpy(), b.numpy()))
        if act != "identity":
            out.append((act,))
    return out


def without_bias_generators(G, layers, inputs):
    """The oracle's generators (rows) without the zero ones of its linear layers (CORALean adds
    the bias as a box of radius zero): the inputs', then each activation's."""
    keep, h, n = list(range(inputs)), inputs, 0
    for layer in layers:
        n = layer[1].shape[0] if layer[0] == "linear" else n
        if layer[0] != "linear":
            keep += list(range(h, h + n))
        h += n
    return G[keep]


def same_bits(test, a, b, what):
    """a and b as float64 arrays of one shape and the same bits."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    test.assertEqual(a.shape, b.shape, what)
    test.assertTrue(np.array_equal(a.view(np.uint64), b.view(np.uint64)), f"{what}: {a} != {b}")


@contextlib.contextmanager
def lean_nearest():
    """The lean backend at dtype nearest, where CORA.cpp's answer is the oracle's bit for bit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cora.CORAWarning)  # nearest has no guarantee, by design
        with cora.backend("lean"), cora.lean.dtype("nearest"):
            yield


def check_zonotope(test, layers, c, r, what):
    """CORA.cpp's output zonotope of the lowered layers on the box (c, r) against the oracle's."""
    active = np.flatnonzero(r > 0)
    G = np.zeros((len(active), len(c)))
    G[np.arange(len(active)), active] = r[active]
    with lean_nearest():
        nn = cora.NeuralNetwork([(W.numpy(), b.numpy()) for W, b in layers[0]], layers[1])
        Y = nn.evaluate(cora.Zonotope(c, G))
        wire = lean_layers(layers)
        L = cora.lean.evaluate_network(wire, cora.lean.Zonotope(c[:, None], G.T), "singh")
        nominal, error = L.gather()
        same_bits(test, Y.center, nominal.center, what + " c")
        theirs = without_bias_generators(np.asarray(nominal.generators, dtype=np.float64), wire, len(active))
        same_bits(test, Y.generators, theirs, what + " G")
        same_bits(test, error.inf, np.zeros(len(error.inf)), what + " no error box")
    return np.asarray(nominal.center, dtype=np.float64).reshape(-1), theirs


def acas_sized(path, seed):
    """A ReLU network 5-24-24-5 of random weights (ACAS Xu's shape, narrower) saved to path."""
    rng = np.random.default_rng(seed)
    sizes, nodes, init, h = [5, 24, 24, 5], [], {}, "x"
    for k in range(3):
        init[f"W{k}"] = (rng.standard_normal((sizes[k + 1], sizes[k])) / np.sqrt(sizes[k])).astype(np.float32)
        init[f"b{k}"] = (0.1 * rng.standard_normal(sizes[k + 1])).astype(np.float32)
        nodes.append(helper.make_node("Gemm", [h, f"W{k}", f"b{k}"], [f"g{k}"], transB=1))
        h = f"g{k}"
        if k < 2:
            nodes.append(helper.make_node("Relu", [h], [f"r{k}"]))
            h = f"r{k}"
    nodes[-1].output[0] = "y"
    onnx.save(model(nodes, [("x", [1, 5])], [("y", [1, 5])], init), path)


# ============================================  MAIN  ============================================ #

@needs_oracle
class Layers(unittest.TestCase):
    def test_every_lowered_layer_type(self):
        # the dense layers of both graphs, on a box and on a box with one fixed input
        for m in (dense_graph(), conv_graph()):
            layers, why = OnnxNetwork(m).dense_layers()
            self.assertIsNone(why)
            n = layers[0][0][0].shape[1]
            for fixed in (False, True):
                c, r = RNG.uniform(-0.5, 0.5, n), RNG.uniform(0.05, 0.4, n)
                r[0] = 0.0 if fixed else r[0]
                check_zonotope(self, layers, c, r, f"{m.graph.node[0].op_type} fixed={fixed}")


@needs_oracle
class EndToEnd(unittest.TestCase):
    def test_three_verdicts(self):
        # far below the range of Y_0 unsat, inside sat, just below unknown (one batch)
        with tempfile.TemporaryDirectory() as folder:
            path, prop = os.path.join(folder, "net.onnx"), os.path.join(folder, "p.vnnlib")
            acas_sized(path, 2)
            net = OnnxNetwork.load(path, dtype=torch.float64)
            y0 = net(torch.as_tensor(np.random.default_rng(0).uniform(-0.1, 0.1, (4000, 5))))[:, 0]
            low = float(y0.min())
            layers = net.dense_layers()[0]
            # the first set of the bounding, the input box, through the oracle
            check_zonotope(self, layers, np.zeros(5), np.full(5, 0.1), "the input box")
            verdicts = []
            for threshold, timeout, max_iter in ((low - 1.0, 60, 10**9), (low + 0.01, 60, 10**9),
                                                 (low - 1e-3, 5, 1)):
                lines = [f"(declare-const X_{i} Real)" for i in range(5)] + \
                    [f"(declare-const Y_{j} Real)" for j in range(5)]
                lines += [f"(assert (<= X_{i} 0.1))\n(assert (>= X_{i} -0.1))" for i in range(5)]
                lines += [f"(assert (<= Y_0 {threshold!r}))"]
                with open(prop, "w") as out:
                    out.write("\n".join(lines) + "\n")
                # CORA's preset on lean (float64, the oracle's order of Fourier-Motzkin)
                options = cora.vnncomp_options("test", dtype="float64", max_verif_iter=max_iter,
                                               falsification_order="bound-first",
                                               falsification_method="center",
                                               fourier_motzkin_order="coralean")
                with lean_nearest():
                    instance = nn_verify.load(path, prop)
                    verdicts.append(nn_verify.verify(instance, timeout, options).status)
        self.assertEqual(verdicts, ["unsat", "sat", "unknown"])

# ----------------------------------------  END OF CODE  ----------------------------------------- #
