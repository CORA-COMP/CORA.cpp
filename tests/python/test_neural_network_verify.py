"""test_neural_network_verify - NeuralNetwork.verify from Python: verdicts, options, the callback.

CORA authors: Lukas Koller, Tobias Ladner
Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: nn/@neuralNetwork/verify.m
# CORA: nn/@neuralNetwork/computePGDAttack.m
# CORA: nn/@neuralNetwork/computeFGSMAttack.m
# CORA: nn/@neuralNetwork/evaluateZonotopeBatch_.m
# CORA: nn/@neuralNetwork/prepareForZonoBatchEval.m
# CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

RNG = np.random.default_rng(4)
LAYERS = [(RNG.standard_normal((8, 2)) / 1.5, 0.3 * RNG.standard_normal(8)),
          (RNG.standard_normal((2, 8)) / 3.0, 0.3 * RNG.standard_normal(2))]
NET = cora.NeuralNetwork(LAYERS)
LO, HI = np.full((1, 2), -0.5), np.full((1, 2), 0.5)
A = np.array([[[1.0, 0.0]]])


def least_output():
    """The least sampled y_0 over the box."""
    x = RNG.uniform(-0.5, 0.5, (4000, 2))
    return float(NET.evaluate(x)[:, 0].min())


# ============================================  MAIN  ============================================ #


class Verify(unittest.TestCase):
    def test_verdicts(self):
        # far below the outputs: unsat; around the least output: sat, the output inside
        low = least_output()
        for options in (cora.vnncomp_options("test", dtype="float64"),
                        cora.NnVerifyOptions(falsification_order="bound-first")):
            safe = NET.verify(LO, HI, [0], A, np.array([[low - 1.0]]), 10, options)
            self.assertEqual(safe["status"], "unsat")
            self.assertGreater(safe["stats"]["boxes"], 0)
            unsafe = NET.verify(LO, HI, [0], A, np.array([[low + 0.01]]), 10, options)
            self.assertEqual(unsafe["status"], "sat")
            self.assertLessEqual(unsafe["y"][0], low + 0.01)
            self.assertTrue(np.all(np.abs(unsafe["x"]) <= 0.5))

    def test_confirm_callback(self):
        # the callback decides: a check that refuses everything leaves the attack unknown
        calls = []

        def refuse(x):
            calls.append(np.asarray(x).copy())
            return None

        options = cora.NnVerifyOptions(falsification_method="center", bounding_method="zonotope",
                                       max_verif_iter=0)
        result = NET.verify(LO, HI, [0], A, np.array([[100.0]]), 2, options, refuse)
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(calls and calls[0].shape == (2,))

    def test_options(self):
        # CORA's preset with a keyword, then the refusals
        o = cora.vnncomp_options("cora_2024", num_approx_err=7)
        self.assertEqual(o.num_neuron_splits, 1)
        self.assertTrue(math.isinf(o.num_relu_constraints))
        self.assertEqual(o.num_approx_err, 7)
        self.assertEqual(o.dtype, "float32")
        with self.assertRaises(cora.CORAError):
            cora.NnVerifyOptions(nope=1)
        with self.assertRaises(cora.CORAError):
            cora.NnVerifyOptions(falsification_method="nope").validate()
        with self.assertRaises(cora.CORAError):
            cora.NnVerifyOptions(dtype="float16")
        # the registered names of the built-ins
        names = cora.verify_strategies()
        self.assertEqual(len(names["heuristics"]), 10)
        self.assertIn("zonotack", names["attacks"])
        self.assertIn("conzonotope", names["boundings"])

    def test_wrong_inputs(self):
        with self.assertRaises(cora.CORAError):
            NET.verify(np.zeros((1, 3)), np.ones((1, 3)), [0], A, np.array([[0.0]]), 1)
        with self.assertRaises(cora.CORAError):
            NET.verify(LO, HI, [1], A, np.array([[0.0]]), 1)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
