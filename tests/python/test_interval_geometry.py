"""test_interval_geometry - rad, radius, partition, project_high_dim, reduce and representsa in Python."""
# CORA: contSet/@interval/partition.m
# CORA: contSet/@interval/projectHighDim_.m
# CORA: contSet/@interval/rad.m
# CORA: contSet/@interval/radius.m
# CORA: contSet/@interval/reduce.m
# CORA: contSet/@interval/representsa_.m
# CORA: contSet/@contSet/projectHighDim.m
# CORA: contSet/@contSet/representsa.m
# CORA: contSet/@contSet/representsa_emptyObject.m

import unittest

import cora
import numpy as np

INF = float("inf")


def box(lo, hi):
    return cora.Interval(np.array(lo, dtype=float), np.array(hi, dtype=float))


class Geometry(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_rad_and_radius(self):
        I = box([0.0, 1.0], [2.0, 9.0])
        np.testing.assert_allclose(I.rad, [1.0, 4.0])
        self.assertAlmostEqual(I.radius(), 17.0 ** 0.5)

    def test_partition(self):
        pieces = box([0.0, 0.0], [4.0, 6.0]).partition(2)
        self.assertEqual(len(pieces), 4)
        self.assertEqual(pieces[1].inf.tolist(), [0.0, 3.0])
        self.assertEqual(pieces[2].sup.tolist(), [4.0, 3.0])
        with self.assertRaises(cora.errors.WrongValue):
            box([0.0], [1.0]).partition(0)

    def test_project_high_dim_and_reduce(self):
        I = box([1.0, 3.0], [2.0, 4.0])
        R = I.project_high_dim(4, [2, 0])
        self.assertEqual(R.inf.tolist(), [3.0, 0.0, 1.0, 0.0])
        self.assertEqual(R.sup.tolist(), [4.0, 0.0, 2.0, 0.0])
        self.assertEqual(cora.Interval.empty(2).project_high_dim(5, [0, 3]).dim, 5)
        with self.assertRaises(cora.errors.WrongValue):
            I.project_high_dim(1, [0, 0])
        self.assertEqual(I.reduce(3).sup.tolist(), I.sup.tolist())
        with self.assertRaises(cora.errors.WrongValue):
            I.reduce(0)

    def test_representsa(self):
        point, full = box([1.0, 2.0], [1.0, 2.0]), box([0.0, 0.0], [1.0, 2.0])
        self.assertTrue(point.representsa("point", tol=0))
        self.assertFalse(full.representsa("point", tol=0))
        self.assertTrue(box([0.0, 0.0], [0.0, 0.0]).representsa("origin"))
        self.assertTrue(full.representsa("zonotope"))
        self.assertFalse(full.representsa("capsule"))
        self.assertTrue(box([0.0, 2.0], [1.0, 2.0]).representsa("capsule"))
        self.assertTrue(cora.Interval.empty(2).representsa("emptySet"))
        self.assertTrue(box([-INF, -INF], [INF, INF]).representsa("fullspace"))
        with self.assertRaises(cora.errors.WrongValue):
            full.representsa("circle")
        with self.assertRaises(cora.errors.NotSupported):
            full.representsa("polygon")


if __name__ == "__main__":
    unittest.main()
