"""test_nn_verify - an ONNX instance through the C++ verifier: falsified, verified, unknown

A ReLU network of 2 inputs with a sat property (PGD finds a counterexample that passes the
scoring's check through the callback), an unsat one (CORA's verify and the zonotope bounding)
and a zero budget (unknown), on the eigen and the torch backend. A network that does not lower to
dense layers (a max pool) is unknown with the reason; one with a layer that C++ cannot take (a
sine) is unknown at once, without reading the property; a mean masked by inputs that the
property fixes lowers once they are folded (nn4sys' mscn), equal to the graph, and falsifies;
masks fixed to other values in other cases fold group by group (unsat only if every group is).
Two networks of a 2.0 query (equal-to with an inequality of inputs, isomorphic-to with equal
inputs) are one joint network: sat with a counterexample per network that the scoring's check
accepts, and unsat; a graph of two inputs is unknown; a property that multiplies an output with an
input is decided by the branch and bound of boxes (sat and unsat).

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   25-November-2026
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

    def test_fail_fast(self):
        # a layer that C++ cannot take: unknown at once, with the reason, the property not read
        path = os.path.join(self.dir.name, "erf.onnx")
        network(path, activation="Erf")
        instance = nn_verify.load(path, os.path.join(self.dir.name, "missing.vnnlib"))
        self.assertIsNone(instance.network)
        self.assertIsNone(instance.prop)
        result = nn_verify.verify(instance, 10)
        self.assertEqual(result.status, "unknown")
        self.assertIn("Erf", result.stats["bounding"])

    def test_fixed_inputs_fold(self):
        # nn4sys' mscn: a mean masked by inputs that the property fixes (the file's docstring)
        path = os.path.join(self.dir.name, "mask.onnx")
        masked_mean_network(path)
        mask = (1.0, 1.0, 0.0)
        box = "".join(
            f"(assert (<= X[0,{i},{j}] 1.0))\n(assert (>= X[0,{i},{j}] -1.0))\n"
            for i in range(3) for j in range(2)
        ) + "".join(
            f"(assert (<= X[0,{i},2] {m}))\n(assert (>= X[0,{i},2] {m}))\n" for i, m in enumerate(mask)
        )
        head = ("(vnnlib-version <2.0>)\n(declare-network N\n (declare-input X float32 [1, 3, 3])\n"
                " (declare-output Y float32 [1, 1])\n)\n")
        prop = os.path.join(self.dir.name, "mask.vnnlib")
        with open(prop, "w") as out:
            out.write(head + box + "(assert (<= Y[0,0] 100.0))\n")
        # it lowers and equals the graph at points of the box
        instance = nn_verify.load(path, prop)
        self.assertIsNotNone(instance.network, instance.why)
        rng = np.random.default_rng(5)
        for _ in range(5):
            x = np.concatenate([rng.uniform(-1, 1, (3, 2)), np.array(mask)[:, None]], 1)
            want = masked_mean(x)
            got = np.asarray(instance.network.evaluate(x.reshape(1, -1))).ravel()
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        # the attack's counterexample passes the scoring's check
        result = nn_verify.verify(instance, 20, cora.vnncomp_options("test"))
        self.assertEqual(result.status, "sat")
        check, why = counterexample.check_counterexample(path, prop, result.counterexample)
        self.assertEqual(check, counterexample.CORRECT, why)
        # a free mask: no constant to fold, unknown with the reason
        with open(prop, "w") as out:
            out.write(head + box.replace("(>= X[0,2,2] 0.0)", "(>= X[0,2,2] -1.0)")
                      + "(assert (<= Y[0,0] 100.0))\n")
        self.assertIsNone(nn_verify.load(path, prop).network)


    def test_fixed_inputs_fold_in_groups(self):
        # mscn's masks fixed to other values in other cases: one folded network per mask
        path = os.path.join(self.dir.name, "mask.onnx")
        masked_mean_network(path)
        masks = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]

        def case(mask, k):
            # the features in [-1, 1] (one fixed to k / 4), the mask fixed
            bounds = [(-1.0, 1.0)] * 6
            bounds[0] = (k / 4, k / 4)
            bounds = [bounds[2 * i + j] if j < 2 else (m, m) for i, m in enumerate(mask)
                      for j in range(3)]
            return "(and " + " ".join(f"(>= X_{i} {lo}) (<= X_{i} {hi})"
                                       for i, (lo, hi) in enumerate(bounds)) + ")"

        def write(rhs):
            prop = os.path.join(self.dir.name, "groups.vnnlib")
            with open(prop, "w") as out:
                out.write("".join(f"(declare-const X_{i} Real)\n" for i in range(9))
                          + "(declare-const Y_0 Real)\n(assert (or "
                          + " ".join(case(m, k) for k, m in enumerate(masks)) + "))\n"
                          + f"(assert (>= Y_0 {rhs}))\n")
            return prop

        bound = float(np.abs(WO).ravel() @ np.abs(WF).sum(0)) + 1.0  # |y| < bound on the boxes
        prop = write(bound)
        instance = nn_verify.load(path, prop)
        self.assertIsNotNone(instance.network, instance.why)
        self.assertEqual([cases for _, cases in instance.parts], [[0, 2], [1]])
        rng = np.random.default_rng(6)
        for network, cases in instance.parts:
            for k in cases:
                for _ in range(3):
                    x = np.concatenate([rng.uniform(-1, 1, (3, 2)), np.array(masks[k])[:, None]], 1)
                    x[0, 0] = k / 4
                    got = np.asarray(network.evaluate(x.reshape(1, -1))).ravel()
                    np.testing.assert_allclose(got, masked_mean(x), rtol=1e-6, atol=1e-6)
        options = cora.vnncomp_options("test")
        result = nn_verify.verify(instance, 30, options)
        self.assertEqual(result.status, "unsat")
        self.assertEqual(result.stats["parts"], 2)
        # the parts' margin of 1e-5 and float64 search were the call's own
        self.assertEqual((options.verified_margin, options.dtype), (1e-4, "float32"))
        # a reachable output: sat, the counterexample checked on the graph
        prop = write(-bound)
        result = nn_verify.verify(nn_verify.load(path, prop), 30, cora.vnncomp_options("test"))
        self.assertEqual(result.status, "sat")
        check, why = counterexample.check_counterexample(path, prop, result.counterexample)
        self.assertEqual(check, counterexample.CORRECT, why)


class SeveralNetworks(unittest.TestCase):
    """VNN-LIB 2.0's pairs: monotonic_acasxu's equal-to (one file) with X_g[0] <= X_f[0] and
    X_g[1] == X_f[1], isomorphic_acasxu's isomorphic-to (two files) with X_g == X_f."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.f = os.path.join(self.dir.name, "f.onnx")
        self.g = os.path.join(self.dir.name, "g.onnx")
        network(self.f)
        network(self.g, weights=(W1 * 1.1, B1, W2, B2 + 0.05))

    def write(self, relation, unsafe, related="equal-to"):
        """X_f in [-1, 1]^2, the relations of X_g, the unsafe outputs; its path."""
        text = (
            "(vnnlib-version <2.0>)\n"
            "(declare-network f\n (declare-input X_f float32 [1, 2])\n"
            " (declare-output Y_f float32 [1, 2])\n)\n"
            f"(declare-network g\n ({related} f)\n (declare-input X_g float32 [1, 2])\n"
            " (declare-output Y_g float32 [1, 2])\n)\n"
            + "".join(f"(assert (and (<= X_f[0,{i}] 1.0) (>= X_f[0,{i}] -1.0)))\n" for i in range(2))
            + relation + unsafe
        )
        path = os.path.join(self.dir.name, "pair.vnnlib")
        with open(path, "w") as out:
            out.write(text)
        return path

    def test_monotonic(self):
        # min over X_g[0] <= X_f[0] (X_g[0] >= -1) of Y_g[0] - Y_f[0], X_g[1] = X_f[1], on a grid
        g = np.linspace(-1, 1, 81)
        a, c, b = np.meshgrid(g, g, g, indexing="ij")
        keep = b <= a
        xf = np.stack([a[keep], c[keep]], 1)
        xg = np.stack([b[keep], c[keep]], 1)
        low = (outputs(xg) - outputs(xf))[:, 0].min()
        relation = ("(assert (and (>= X_f[0,0] X_g[0,0]) (>= X_g[0,0] -1.0)))\n"
                    "(assert (== X_f[0,1] X_g[0,1]))\n")
        field = str([("f", self.f), ("g", self.f)])
        sat = self.write(relation, f"(assert (<= (- Y_g[0,0] Y_f[0,0]) {float(low) + 0.05!r}))\n")
        result = nn_verify.verify(nn_verify.load(field, sat), 30, cora.vnncomp_options("test"))
        self.assertEqual(result.status, "sat")
        check, why = counterexample.check_counterexample(field, sat, result.counterexample)
        self.assertEqual(check, counterexample.CORRECT, why)
        x = np.asarray(result.x, dtype=np.float64)
        self.assertLessEqual(x[2], x[0])
        self.assertEqual(x[1], x[3])
        # the counterexample per network: X_f, Y_f, X_g, Y_g
        heads = [line.split()[0] for line in result.counterexample.splitlines() if "[" in line]
        self.assertEqual(heads, ["X_f", "Y_f", "X_g", "Y_g"])
        unsat = self.write(relation, f"(assert (<= (- Y_g[0,0] Y_f[0,0]) {float(low) - 0.5!r}))\n")
        result = nn_verify.verify(nn_verify.load(field, unsat), 30, cora.vnncomp_options("test"))
        self.assertEqual(result.status, "unsat")

    def test_isomorphic(self):
        # X_g == X_f, two files: the largest Y_g[0] - Y_f[0] on a grid
        relation = "".join(f"(assert (== X_f[0,{i}] X_g[0,{i}]))\n" for i in range(2))
        g = np.linspace(-1, 1, 401)
        x = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        gap = (outputs(x, (W1 * 1.1, B1, W2, B2 + 0.05)) - outputs(x))[:, 0]
        field = str([("f", self.f), ("g", self.g)])
        for rhs, want in ((float(gap.max()) - 0.05, "sat"), (float(gap.max()) + 0.5, "unsat")):
            path = self.write(relation, f"(assert (>= (- Y_g[0,0] Y_f[0,0]) {rhs!r}))\n",
                              "isomorphic-to")
            instance = nn_verify.load(field, path)
            self.assertEqual(instance.network.input_dim, 2)  # X_g read from X_f
            result = nn_verify.verify(instance, 30, cora.vnncomp_options("test"))
            self.assertEqual(result.status, want)
            if want == "sat":
                check, why = counterexample.check_counterexample(field, path, result.counterexample)
                self.assertEqual(check, counterexample.CORRECT, why)
                # the same counterexample with X_g moved breaks X_g == X_f
                moved = result.counterexample.replace(f"X_g float32 [1, 2]\n{float(result.x[2])!r}",
                                                      "X_g float32 [1, 2]\n0.123", 1)
                self.assertNotEqual(moved, result.counterexample)
                check, _ = counterexample.check_counterexample(field, path, moved)
                self.assertEqual(check, counterexample.SPEC_NOT_VIOLATED)

    def test_several_inputs_unknown(self):
        # a graph of two inputs: unknown with the reason, the property not read
        path = os.path.join(self.dir.name, "two.onnx")
        two_input_network(path)
        instance = nn_verify.load(path, os.path.join(self.dir.name, "missing.vnnlib"))
        result = nn_verify.verify(instance, 5)
        self.assertEqual(result.status, "unknown")
        self.assertIn("one input", result.stats["bounding"])


class Nonlinear(unittest.TestCase):
    def test_products_of_inputs_and_outputs(self):
        # Y_0 X_0 >= c: sat below the grid's largest value, unsat above it
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "net.onnx")
            network(onnx_path)
            g = np.linspace(-1, 1, 401)
            x = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
            best = float((outputs(x)[:, 0] * x[:, 0]).max())
            # each property through the branch and bound of the boxes
            for rhs, want in ((best - 0.05, "sat"), (best + 0.5, "unsat")):
                path = os.path.join(tmp, "nl.vnnlib")
                with open(path, "w") as out:
                    out.write(
                        "(vnnlib-version <2.0>)\n(declare-network N\n"
                        " (declare-input X float32 [1, 2])\n (declare-output Y float32 [1, 2])\n)\n"
                        + "".join(f"(assert (<= X[0,{i}] 1.0))\n(assert (>= X[0,{i}] -1.0))\n"
                                  for i in range(2))
                        + f"(assert (>= (* Y[0,0] X[0,0]) {rhs!r}))\n")
                instance = nn_verify.load(onnx_path, path)
                self.assertIsNotNone(instance.nonlinear)
                result = nn_verify.verify(instance, 30)
                self.assertEqual(result.status, want)
                self.assertEqual(result.stats["bounding"], "nonlinear-bab")
                if want == "sat":
                    check, why = counterexample.check_counterexample(onnx_path, path,
                                                                     result.counterexample)
                    self.assertEqual(check, counterexample.CORRECT, why)


class Precision(unittest.TestCase):
    def test_tf32_off_at_first_torch_tensor(self):
        # a fresh process: the first torch tensor handed to cora turns TensorFloat-32 off
        import subprocess
        import sys

        code = (
            "import torch, cora\n"
            "torch.backends.cudnn.allow_tf32 = True\n"
            "torch.backends.cuda.matmul.allow_tf32 = True\n"
            "cora.Zonotope(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))\n"
            "print(torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ), check=True).stdout.split()
        self.assertEqual(out[-2:], ["False", "False"])


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

RNG = np.random.default_rng(11)

W1, B1 = RNG.standard_normal((16, 2)).astype(np.float32), RNG.standard_normal(16).astype(np.float32)

W2, B2 = RNG.standard_normal((2, 16)).astype(np.float32), RNG.standard_normal(2).astype(np.float32)


def network(path, activation="Relu", weights=None):
    """y = W2 relu(W1 x + b1) + b2 for x of shape (1, 2) (or another activation, other weights)."""
    w1, b1, w2, b2 = (np.asarray(w, dtype=np.float32) for w in (weights or (W1, B1, W2, B2)))
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], transB=1),
        helper.make_node(activation, ["h"], ["r"]),
        helper.make_node("Gemm", ["r", "W2", "b2"], ["y"], transB=1),
    ]
    init = [numpy_helper.from_array(a, n) for n, a in (("W1", w1), ("b1", b1), ("W2", w2), ("b2", b2))]
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


WF = RNG.standard_normal((2, 4)).astype(np.float32)

WO = RNG.standard_normal((1, 4)).astype(np.float32)


def masked_mean_network(path):
    """mscn's set mean: X (1, 3, 3) split into features (1, 3, 2) and a mask (1, 3, 1);
    y = WO (sum_i mask_i relu(f_i WF)) / sum_i mask_i."""
    # the mean of the masked features, then one output
    nodes = [
        helper.make_node("Split", ["x"], ["f", "m"], axis=-1, split=[2, 1]),
        helper.make_node("MatMul", ["f", "WF"], ["h"]),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Mul", ["r", "m"], ["rm"]),
        helper.make_node("ReduceSum", ["rm"], ["s"], axes=[1], keepdims=0),
        helper.make_node("ReduceSum", ["m"], ["c"], axes=[1], keepdims=0),
        helper.make_node("Div", ["s", "c"], ["mean"]),
        helper.make_node("Gemm", ["mean", "WO"], ["y"], transB=1),
    ]
    # the graph of opset 11 (Split's sizes an attribute)
    init = [numpy_helper.from_array(WF, "WF"), numpy_helper.from_array(WO, "WO")]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        init,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 7
    onnx.save(m, path)


def masked_mean(x):
    """masked_mean_network on one input (3, 3) in float64."""
    f, m = x[:, :2], x[:, 2:]
    mean = (np.maximum(f @ WF.astype(np.float64), 0) * m).sum(0) / m.sum()
    return WO.astype(np.float64) @ mean


def outputs(x, weights=None):
    """The network on the rows x in float64 (of other weights)."""
    w1, b1, w2, b2 = (np.asarray(w, dtype=np.float32).astype(np.float64)
                      for w in (weights or (W1, B1, W2, B2)))
    return np.maximum(x @ w1.T + b1, 0) @ w2.T + b2


def two_input_network(path):
    """y = a + b of two inputs of shape (1, 2)."""
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "b"], ["y"])],
        "g",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 2]) for n in ("a", "b")],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
