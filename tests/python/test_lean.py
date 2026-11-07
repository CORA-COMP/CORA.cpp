"""test_lean - cora.lean: the exact crossing and the dtype rules; the oracle when CORACPP_ORACLE is set

CORA authors: Mark Wetzlinger
Authors:       Claude Sonnet 5.5, Tobias Ladner
Written:       30-September-2026
Last update:   07-November-2026
Reviewed:      ---
"""
# CORA: contDynamics/@linearSys/reach.m

import os
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

class Crossing(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.lean.dtype("binary64"))

    def test_a_double_crosses_exactly(self):
        x = np.array([[0.1, -2.5], [1e-300, 3.0]])
        np.testing.assert_array_equal(cora.lean.Tensor.from_array(x).gather(), x)

    def test_nan_does_not_enter(self):
        with self.assertRaises(cora.errors.CORAError):
            cora.lean.Tensor.from_array(np.array([[np.nan]]))

    def test_an_unknown_dtype_is_an_error(self):
        with self.assertRaises(ValueError):
            with cora.lean.dtype("float16"):
                pass

    def test_an_unknown_reduction_method_is_an_error(self):
        Z = cora.lean.Zonotope(np.array([1.0, 2.0]), np.eye(2))
        with self.assertRaises(ValueError):
            Z.reduce(1, "pca")


@unittest.skipUnless(os.environ.get("CORACPP_ORACLE"), "needs CORACPP_ORACLE")


class Oracle(unittest.TestCase):
    def test_methods_take_arrays_and_return_lean_objects(self):
        # methods take arrays
        self.enterContext(cora.lean.dtype("binary64"))
        Z = cora.lean.Zonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 2.0]]))
        self.assertEqual(Z.support_function(np.array([1.0, 1.0])), 7.5)
        halves = Z.split(1)
        self.assertEqual(len(halves), 2)
        np.testing.assert_array_equal(Z.linear_map(np.eye(2)).gather()[0].center, [1.0, 2.0])
        box = cora.lean.Interval(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
        self.assertEqual(box.volume(), 2.0)
        inner = cora.lean.Interval(np.array([0.25, 0.5]), np.array([0.5, 1.0]))
        self.assertTrue(box.contains(inner).proved)
        self.assertIsNotNone(box.boundary_point(np.array([1.0, 1.0]), np.array([0.5, 0.5])))

    def test_the_lean_reach(self):
        self.enterContext(cora.lean.dtype("binary64"))
        A = np.array([[0.0, 1.0], [-1.0, -0.25]])
        c, G = np.array([1.0, 0.0]), 0.25 * np.eye(2)
        R = cora.lean.LinearSys(A).reach(cora.lean.Zonotope(c, G), 1.0, time_step=0.125, taylor_terms=8)
        self.assertEqual(len(R.time_point), 9)
        self.assertIsInstance(R.time_point[0], cora.lean.Zonotope)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
