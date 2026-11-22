"""test_nn_verify - an ONNX instance through the C++ verifier: falsified, verified, unknown

A ReLU network of 2 inputs with a sat property (PGD finds a counterexample that passes the
scoring's check through the callback), an unsat one (CORA's verify and the zonotope bounding)
and a zero budget (unknown), on the eigen and the torch backend. A network that does not lower to
dense layers (a max pool) is unknown with the reason.

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import os
import tempfile
import unittest


try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401
    import torch  # noqa: F401  the torch backend of the second run
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    raise unittest.SkipTest("needs onnx, onnxruntime and torch") from None

import cora


if not cora.has_torch:
    raise unittest.SkipTest("needs a build with torch") from None

import cora.counterexample as counterexample
import cora.nn_verify as nn_verify


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Verdicts(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.onnx = os.path.join(self.dir.name, "net.onnx")
        network(self.onnx)
        # the extremes of Y_0 - Y_1 over a fine grid decide the two properties
        g = np.linspace(-1, 1, 401)
        x = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        d = outputs(x) @ np.array([1.0, -1.0])
        self.low = d.min()

    def property(self, name, rhs):
        """X in [-1, 1]^2 and the unsafe set Y_0 - Y_1 <= rhs (a 2.0 text)."""
        text = (
            "(vnnlib-version <2.0>)\n(declare-network N\n (declare-input X float32 [1, 2])\n"
            " (declare-output Y float32 [1, 2])\n)\n"
            + "".join(f"(assert (<= X[0,{i}] 1.0))\n(assert (>= X[0,{i}] -1.0))\n" for i in range(2))
            + f"(assert (<= (+ (* 1.0 Y[0,0]) (* -1.0 Y[0,1])) {rhs!r}))\n"
        )
        path = os.path.join(self.dir.name, name)
        with open(path, "w") as out:
            out.write(text)
        return path

    def test_falsified(self):
        # PGD finds a point that the scoring's check accepts, on both backends
        path = self.property("sat.vnnlib", float(self.low) + 0.05)
        for backend in ("eigen", "torch"):
            result = nn_verify.verify(nn_verify.load(self.onnx, path, backend=backend), 30,
                                      cora.vnncomp_options("test"))
            self.assertEqual(result.status, "sat", backend)
            check, why = counterexample.check_counterexample(self.onnx, path, result.counterexample)
            self.assertEqual(check, counterexample.CORRECT, why)

    def test_verified(self):
        # both boundings and both orders verify a set below the minimum
        path = self.property("unsat.vnnlib", float(self.low) - 0.5)
        instance = nn_verify.load(self.onnx, path)
        for options in (cora.vnncomp_options("test"),
                        cora.NnVerifyOptions(falsification_order="bound-first")):
            result = nn_verify.verify(instance, 60, options)
            self.assertEqual(result.status, "unsat", options.bounding_method)
            self.assertGreater(result.stats["boxes"], 0)

    def test_unknown(self):
        # no time: unknown; a graph that is no dense chain: unknown with the reason
        path = self.property("hard.vnnlib", float(self.low) - 0.5)
        self.assertEqual(nn_verify.verify(nn_verify.load(self.onnx, path), 0.0).status, "unknown")
        instance = nn_verify.load(self.onnx, path)
        instance.network, instance.why = None, "a max pool"
        result = nn_verify.verify(instance, 10)
        self.assertEqual(result.status, "unknown")
        self.assertIn("max pool", result.stats["bounding"])


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

RNG = np.random.default_rng(11)

W1, B1 = RNG.standard_normal((16, 2)).astype(np.float32), RNG.standard_normal(16).astype(np.float32)

W2, B2 = RNG.standard_normal((2, 16)).astype(np.float32), RNG.standard_normal(2).astype(np.float32)


def network(path):
    """y = W2 relu(W1 x + b1) + b2 for x of shape (1, 2)."""
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], transB=1),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Gemm", ["r", "W2", "b2"], ["y"], transB=1),
    ]
    init = [numpy_helper.from_array(a, n) for n, a in (("W1", W1), ("b1", B1), ("W2", W2), ("b2", B2))]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        init,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def outputs(x):
    """The network on the rows x in float64."""
    return np.maximum(x @ W1.T.astype(np.float64) + B1, 0) @ W2.T.astype(np.float64) + B2

# ----------------------------------------  END OF CODE  ----------------------------------------- #
