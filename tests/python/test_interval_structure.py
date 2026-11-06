"""test_interval_structure - split, sqrt, sum, supremum, tan, tanh, times, transpose, tril, triu,

uminus, uplus, vertcat, volume and zonotope of an interval in Python.

CORA authors: Niklas Kochdumper, Matthias Althoff, Daniel Althoff, Dmitry Grebenyuk,
              Mark Wetzlinger, Victor Gassmann
Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@interval/split.m
# CORA: contSet/@interval/sqrt.m
# CORA: contSet/@interval/sum.m
# CORA: contSet/@interval/supremum.m
# CORA: contSet/@interval/tan.m
# CORA: contSet/@interval/tanh.m
# CORA: contSet/@interval/times.m
# CORA: contSet/@interval/transpose.m
# CORA: contSet/@interval/tril.m
# CORA: contSet/@interval/triu.m
# CORA: contSet/@interval/uminus.m
# CORA: contSet/@interval/uplus.m
# CORA: contSet/@interval/vertcat.m
# CORA: contSet/@interval/volume_.m
# CORA: contSet/@interval/zonotope.m
# PROPERTY: contSet/@interval/sqrt.m
# PROPERTY: contSet/@interval/tan.m
# PROPERTY: contSet/@interval/tanh.m
# PROPERTY: contSet/@interval/times.m
# PROPERTY: contSet/@interval/zonotope.m

import unittest

import cora
import numpy as np

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

INF = float("inf")


def box(lo, hi):
    return cora.Interval(np.array(lo, dtype=float), np.array(hi, dtype=float))


def matrix(lo, hi):
    return cora.Interval.matrix(np.array(lo, dtype=float), np.array(hi, dtype=float))


# ============================================  MAIN  ============================================ #

class Structure(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_signs_and_bounds(self):
        I = box([-1.0, 0.0], [2.0, 0.5])
        np.testing.assert_array_equal((-I).inf, [-2.0, -0.5])
        np.testing.assert_array_equal((-I).sup, [1.0, 0.0])
        np.testing.assert_array_equal((+I).inf, I.inf)
        np.testing.assert_array_equal(I.supremum(), [2.0, 0.5])

    def test_transpose_and_stacking(self):
        M = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        np.testing.assert_array_equal(M.transpose().inf, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(M.T.sup, M.transpose().sup)
        a, b = box([0.0, 1.0], [1.0, 2.0]), box([5.0], [6.0])
        V = cora.Interval.vertcat([a, b, np.array([9.0, 9.0])[:1]])
        np.testing.assert_array_equal(V.inf, [0.0, 1.0, 5.0, 9.0])
        np.testing.assert_array_equal(V.sup, [1.0, 2.0, 6.0, 9.0])
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.Interval.vertcat([a, M])

    def test_sum_and_triangles(self):
        M = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        np.testing.assert_array_equal(M.sum(1).inf, [[5.0, 7.0, 9.0]])
        np.testing.assert_array_equal(M.sum(dim=2).sup, [9.0, 18.0])
        self.assertEqual(box([1.0, 2.0], [2.0, 3.0]).sum().sup.tolist(), [5.0])
        with self.assertRaises(cora.errors.WrongValue):
            M.sum(3)
        N = matrix(np.arange(9.0).reshape(3, 3), np.arange(9.0).reshape(3, 3) + 1)
        np.testing.assert_array_equal(N.tril().inf, np.tril(np.arange(9.0).reshape(3, 3)))
        np.testing.assert_array_equal(N.triu(1).inf, np.triu(np.arange(9.0).reshape(3, 3), 1))
        with self.assertRaises(cora.errors.WrongValue):
            N.tril(-1)

    def test_volume_and_split(self):
        I = box([1.0, -2.0], [3.0, 4.0])
        self.assertEqual(I.volume(), 12.0)
        self.assertEqual(box([1.0, 1.0], [3.0, 1.0]).volume(), 0.0)
        I1, I2 = box([0.0, -2.0, 1.0], [2.0, 4.0, 5.0]).split(1)
        np.testing.assert_array_equal(I1.sup, [2.0, 1.0, 5.0])
        np.testing.assert_array_equal(I2.inf, [0.0, 1.0, 1.0])
        with self.assertRaises(cora.errors.WrongValue):
            I.split(2)

    def test_functions(self):
        S = box([0.0, 1.0], [4.0, 9.0]).sqrt()
        np.testing.assert_allclose(S.sup, [2.0, 3.0])
        with self.assertRaises(cora.errors.OutOfDomain):
            box([-1.0], [1.0]).sqrt()
        T = box([-1.0, 1.0], [1.0, 3.0]).tan()
        np.testing.assert_allclose(T.sup[:1], [np.tan(1.0)])
        self.assertEqual((T.inf[1], T.sup[1]), (-INF, INF))
        H = box([-1.0], [2.0]).tanh()
        np.testing.assert_allclose([H.inf[0], H.sup[0]], np.tanh([-1.0, 2.0]))

    def test_samples_stay_inside(self):
        rng = np.random.default_rng(5)
        for name, f, lo_, hi_ in (("sqrt", np.sqrt, 0, 5), ("tanh", np.tanh, -3, 3), ("tan", np.tan, -4, 4)):
            lo = rng.uniform(lo_, hi_, 4)
            I = box(lo, lo + rng.uniform(0, 2, 4))
            R = getattr(I, name)()
            x = lo[:, None] + (I.sup - lo)[:, None] * rng.uniform(0, 1, (4, 30))
            y = f(x)
            self.assertTrue(np.all(R.inf[:, None] <= y + 1e-12) and np.all(y <= R.sup[:, None] + 1e-12), name)

    def test_times(self):
        A, B = box([1.0, -2.0], [3.0, 4.0]), box([2.0, 0.0], [5.0, 1.0])
        R = A.times(B)
        np.testing.assert_array_equal(R.inf, [2.0, -2.0])
        np.testing.assert_array_equal(R.sup, [15.0, 4.0])
        N = A.times(np.array([-2.0, 3.0]))
        np.testing.assert_array_equal(N.inf, [-6.0, -6.0])
        np.testing.assert_array_equal(N.sup, [-2.0, 12.0])
        with self.assertRaises(cora.errors.DimensionMismatch):
            A.times(np.array([1.0, 2.0, 3.0]))
        rng = np.random.default_rng(6)
        x = A.inf[:, None] + (A.sup - A.inf)[:, None] * rng.uniform(0, 1, (2, 30))
        y = B.inf[:, None] + (B.sup - B.inf)[:, None] * rng.uniform(0, 1, (2, 30))
        self.assertTrue(np.all(R.inf[:, None] <= x * y) and np.all(x * y <= R.sup[:, None]))

    def test_zonotope(self):
        I = box([0.0, 1.0, -1.0], [2.0, 1.0, 3.0])
        Z = I.zonotope()
        np.testing.assert_array_equal(Z.center, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(Z.generators, [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
        H = Z.interval()
        np.testing.assert_allclose(H.inf, I.inf)
        np.testing.assert_allclose(H.sup, I.sup)
        with self.assertRaises(cora.errors.WrongValue):
            matrix([[0.0, 1.0]], [[1.0, 2.0]]).zonotope()

    def test_batches(self):
        B = cora.Interval.stack([box([-1.0, 0.5], [2.0, 3.0])] * 2)
        self.assertEqual((-B).inf.shape, (2, 2))
        self.assertEqual(B.abs().sqrt().inf.shape, (2, 2))
        self.assertEqual(B.tanh().sup.shape, (2, 2))
        self.assertEqual(B.tan().sup.shape, (2, 2))
        self.assertEqual(B.volume().shape, (2,))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
