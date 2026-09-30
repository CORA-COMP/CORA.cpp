"""test_lean - cora.lean: the exact crossing and the dtype rules; the oracle when CORACPP_ORACLE is set"""
import os
import unittest

import numpy as np

import cora


class Crossing(unittest.TestCase):
    def setUp(self):
        cora.lean.setDType("binary64")

    def test_a_double_crosses_exactly(self):
        x = np.array([[0.1, -2.5], [1e-300, 3.0]])
        np.testing.assert_array_equal(cora.lean.Tensor.fromArray(x).gather(), x)

    def test_nan_does_not_enter(self):
        with self.assertRaises(Exception):
            cora.lean.Tensor.fromArray(np.array([[np.nan]]))

    def test_an_unknown_dtype_is_an_error(self):
        with self.assertRaises(ValueError):
            cora.lean.setDType("float16")

    def test_an_unknown_reduction_method_is_an_error(self):
        Z = cora.lean.Zonotope(np.array([1.0, 2.0]), np.eye(2))
        with self.assertRaises(ValueError):
            Z.reduce(1, "pca")


@unittest.skipUnless(os.environ.get("CORACPP_ORACLE"), "needs the CORALean oracle")
class Oracle(unittest.TestCase):
    def test_the_lean_reach_encloses_the_double_reach(self):
        cora.lean.setDType("binary64")
        A = np.array([[0.0, 1.0], [-1.0, -0.25]])
        c, G = np.array([1.0, 0.0]), 0.25 * np.eye(2)
        R = cora.lean.LinearSys(A).reach(cora.lean.Zonotope(c, G), 0.125, 1.0, 8)
        X0 = cora.Zonotope(cora.Tensor(c), cora.Tensor(G))
        ref = cora.LinearSys(cora.Tensor(A)).reach(X0, 0.125, 1.0, 8)
        self.assertEqual(len(R.timePoint), 9)
        for lean_set, z in zip(R.timePoint, ref.timePoint):
            lo, hi = lean_set.interval().gather().inf, lean_set.interval().gather().sup
            box = z.interval()
            self.assertTrue(np.all(np.asarray(lo) <= np.asarray(box.inf) + 1e-9))
            self.assertTrue(np.all(np.asarray(hi) >= np.asarray(box.sup) - 1e-9))


if __name__ == "__main__":
    unittest.main()
