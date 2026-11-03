"""test_lean - cora.lean: the exact crossing and the dtype rules; the oracle when CORACPP_ORACLE is set"""
# CORA: contDynamics/@linearSys/reach.m  (enclosure against the oracle only, not bit-exact: not yet ORACLE)

import unittest

import cora
import numpy as np
import oracle_testing


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


@oracle_testing.needs_oracle
class Oracle(unittest.TestCase):
    def test_methods_take_arrays_and_return_lean_objects(self):
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

    def test_the_lean_reach_encloses_the_double_reach(self):
        self.enterContext(cora.lean.dtype("binary64"))
        A = np.array([[0.0, 1.0], [-1.0, -0.25]])
        c, G = np.array([1.0, 0.0]), 0.25 * np.eye(2)
        R = cora.lean.LinearSys(A).reach(cora.lean.Zonotope(c, G), 1.0, time_step=0.125, taylor_terms=8)
        X0 = cora.Zonotope(cora.Tensor(c), cora.Tensor(G))
        ref = cora.LinearSys(cora.Tensor(A)).reach(X0, time_step=0.125, t_final=1.0, taylor_terms=8)
        self.assertEqual(len(R.time_point), 9)
        for k, (lean_set, z) in enumerate(zip(R.time_point, ref.time_point, strict=True)):
            oracle_testing.check_enclosed(f"time point {k}", z.interval(), lean_set.interval())


if __name__ == "__main__":
    unittest.main()
