"""test_nn_verify_memory - an ONNX instance through the C++ verifier with too little memory

A ReLU network of 2 inputs and an unsat property: with budgets of one byte (any backend) and, on
a GPU, with the process capped at a few MiB more than it holds (torch's memory fraction: a GPU
without room for the verifier's sets), verify returns a sound answer, unsat or unknown, instead of
raising an out of memory; the cap lifted, the same instance is unsat again.

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""

import os
import tempfile
import unittest


try:
    import numpy as np
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    raise unittest.SkipTest("needs onnx and torch") from None

import cora


if not cora.has_torch:
    raise unittest.SkipTest("needs a build with torch") from None

import cora.nn_verify as nn_verify


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class TooLittleMemory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.onnx = os.path.join(self.dir.name, "net.onnx")
        network(self.onnx)
        g = np.linspace(-1, 1, 401)
        x = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        self.low = float((outputs(x) @ np.array([1.0, -1.0])).min())
        self.prop = os.path.join(self.dir.name, "unsat.vnnlib")
        with open(self.prop, "w") as out:
            out.write(
                "(vnnlib-version <2.0>)\n(declare-network N\n (declare-input X float32 [1, 2])\n"
                " (declare-output Y float32 [1, 2])\n)\n"
                + "".join(f"(assert (<= X[0,{i}] 1.0))\n(assert (>= X[0,{i}] -1.0))\n"
                          for i in range(2))
                + f"(assert (<= (+ (* 1.0 Y[0,0]) (* -1.0 Y[0,1])) {self.low - 0.3!r}))\n"
            )

    def options(self, device="", **budgets):
        # crown-bab's preset (cifar100), bound first, a short attack
        return cora.vnncomp_options("cifar100", device=device, falsification_order="bound-first",
                                    pgd_restarts=8, pgd_steps=20, **budgets)

    def test_tiny_budgets(self):
        # budgets of one byte: the smallest root net, chunks and batches of one, still sound
        tiny = dict(memory_budget=1, batch_memory_budget=1, relu_bab_memory_budget=1,
                    crown_bab_memory_budget=1)
        for backend in ("eigen", "torch"):
            instance = nn_verify.load(self.onnx, self.prop, backend=backend)
            result = nn_verify.verify(instance, 5, self.options(**tiny))
            self.assertIn(result.status, ("unsat", "unknown"), backend)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_capped_gpu(self):
        # the process capped just above what it holds: out of memory everywhere, unknown or unsat
        instance = nn_verify.load(self.onnx, self.prop, backend="torch", device="cuda")
        options = self.options(device="cuda", batch_memory_budget=1 << 30)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _, total = torch.cuda.mem_get_info()
        cap = (torch.cuda.memory_reserved() + (4 << 20)) / total
        torch.cuda.set_per_process_memory_fraction(cap)
        try:
            capped = nn_verify.verify(instance, 5, options)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0)
            torch.cuda.empty_cache()
        self.assertIn(capped.status, ("unsat", "unknown"))
        self.assertTrue(any(k.endswith("_oom") for k in capped.stats), capped.stats)
        # the cap lifted: unsat
        self.assertEqual(nn_verify.verify(instance, 30, options).status, "unsat")


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


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
