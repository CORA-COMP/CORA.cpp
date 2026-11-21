"""test_nn_verify - verify on small instances: falsified, verified, unknown; the strategy options

A ReLU network of 2 inputs with a sat property (PGD finds a counterexample that passes the
scoring's check), an unsat one (the zonotope bounds and input splitting verify it) and a zero
budget (unknown). The zonotope's lower bounds lie below A y - b at sampled points. Both orders of
falsification and bounding give the same verdicts; an attack registered from outside is chosen by
its name; unknown names and the Phase 1 stubs are refused.

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
    import torch
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    raise unittest.SkipTest("needs onnx, onnxruntime and torch") from None

import cora


if not cora.has_torch:
    raise unittest.SkipTest("needs a build with torch") from None

from cora import counterexample, nn_attack, nn_verify


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

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


# ============================================  MAIN  ============================================ #


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
        self.low, self.high = d.min(), d.max()

    def write(self, name, text):
        path = os.path.join(self.dir.name, name)
        with open(path, "w") as out:
            out.write(text)
        return path

    def property(self, name, row, rhs):
        """X in [-1, 1]^2 and the unsafe set row . y <= rhs (a 2.0 text)."""
        a, b = row
        text = (
            "(vnnlib-version <2.0>)\n(declare-network N\n (declare-input X float32 [1, 2])\n"
            " (declare-output Y float32 [1, 2])\n)\n"
            + "".join(f"(assert (<= X[0,{i}] 1.0))\n(assert (>= X[0,{i}] -1.0))\n" for i in range(2))
            + f"(assert (<= (+ (* {a} Y[0,0]) (* {b} Y[0,1])) {rhs!r}))\n"
        )
        return self.write(name, text)

    def test_falsified(self):
        # unsafe: Y_0 - Y_1 <= its grid minimum + a little: PGD finds a point
        path = self.property("sat.vnnlib", (1.0, -1.0), float(self.low) + 0.05)
        problem = nn_verify.NnProblem.load(self.onnx, path)
        result = nn_verify.verify(problem, timeout=30)
        self.assertEqual(result.status, "sat")
        check, why = counterexample.check_counterexample(self.onnx, path, result.counterexample)
        self.assertEqual(check, counterexample.CORRECT, why)

    def test_verified(self):
        # unsafe: Y_0 - Y_1 <= below its minimum: the bounds and the splits exclude it
        for order in ("attack-first", "bound-first"):
            path = self.property("unsat.vnnlib", (1.0, -1.0), float(self.low) - 0.5)
            options = nn_verify.VerifyOptions(falsification_order=order, attack_time_share=0.05)
            result = nn_verify.verify(nn_verify.NnProblem.load(self.onnx, path, options), timeout=60)
            self.assertEqual(result.status, "unsat", order)
            self.assertGreater(result.stats["boxes"], 1)

    def test_unknown_without_time(self):
        path = self.property("hard.vnnlib", (1.0, -1.0), float(self.low) - 0.5)
        result = nn_verify.verify(nn_verify.NnProblem.load(self.onnx, path), timeout=0.0)
        self.assertEqual(result.status, "unknown")

    def test_bounds_enclose(self):
        # the zonotope's lower bound of A y - b on sub-boxes, below the sampled values
        path = self.property("p.vnnlib", (1.0, -1.0), 0.0)
        problem = nn_verify.NnProblem.load(self.onnx, path)
        # two boxes in one batch
        bounding = nn_verify.ZonotopeBounding()
        self.assertTrue(bounding.prepare(problem))
        lo = torch.tensor([[-1.0, -1.0], [0.0, -0.5]], dtype=torch.float64)
        hi = torch.tensor([[1.0, 1.0], [0.5, 0.25]], dtype=torch.float64)
        lower = bounding.lower(lo, hi, torch.zeros(2, dtype=torch.int64))
        for k in range(2):
            x = lo[k].numpy() + RNG.uniform(0, 1, (500, 2)) * (hi[k] - lo[k]).numpy()
            values = outputs(x) @ np.array([1.0, -1.0])
            self.assertLessEqual(float(lower[k, 0, 0]), values.min())


class Strategies(unittest.TestCase):
    def test_unknown_names(self):
        # every strategy name is checked
        for field in (
            "falsification_method",
            "bounding_method",
            "input_split_heuristic",
            "falsification_order",
        ):
            with self.assertRaises(ValueError, msg=field):
                nn_verify.VerifyOptions(**{field: "nonsense"}).validate()
        with self.assertRaises(NotImplementedError):
            nn_verify.VerifyOptions(neuron_split_heuristic="most-sensitive-approx-error").validate()

    def test_registered_attack_is_used(self):
        # an attack registered from outside, chosen by its name
        calls = []

        @nn_attack.register_attack("test-nothing")
        class Nothing(nn_attack.Attack):
            def run(self, problem, deadline, rounds=None):
                calls.append(problem.num_boxes)
                return nn_attack.AttackResult()

        # an instance that the bounds verify; the attack runs once on its one box
        self.addCleanup(nn_attack.ATTACKS.pop, "test-nothing")
        with tempfile.TemporaryDirectory() as folder:
            onnx_path = os.path.join(folder, "net.onnx")
            network(onnx_path)
            path = os.path.join(folder, "p.vnnlib")
            with open(path, "w") as out:
                out.write(
                    "".join(f"(declare-const {v} Real)\n" for v in ("X_0", "X_1", "Y_0", "Y_1"))
                    + "(assert (<= X_0 1))\n(assert (>= X_0 -1))\n(assert (<= X_1 1))\n"
                    "(assert (>= X_1 -1))\n(assert (>= Y_0 1e9))\n"
                )
            options = nn_verify.VerifyOptions(falsification_method="test-nothing")
            result = nn_verify.verify(nn_verify.NnProblem.load(onnx_path, path, options), timeout=30)
        self.assertEqual(result.status, "unsat")
        self.assertEqual(calls, [1])

    def test_stubs(self):
        with self.assertRaises(NotImplementedError):
            nn_attack.make_attack("zonotack").run(None, 0)
        with self.assertRaises(NotImplementedError):
            nn_verify.BOUNDING_METHODS["conzonotope"]().prepare(None)


# ----------------------------------------  END OF CODE  ----------------------------------------- #
