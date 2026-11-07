"""test_interval_functions - sin, sinh, sign, round, power, rdivide and prod of an interval in Python.

CORA authors: Dmitry Grebenyuk, Tobias Ladner, Sebastian Mair, Mark Wetzlinger, Matthias Althoff
Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@interval/power.m
# CORA: contSet/@interval/prod.m
# CORA: contSet/@interval/rdivide.m
# CORA: contSet/@interval/round.m
# CORA: contSet/@interval/sign.m
# CORA: contSet/@interval/sin.m
# CORA: contSet/@interval/sinh.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

INF = float("inf")


def box(lo, hi):
    return cora.Interval(np.array(lo, dtype=float), np.array(hi, dtype=float))


# ============================================  MAIN  ============================================ #

class Functions(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_sin_and_sinh(self):
        R = box([0.0, -math.pi / 2, 0.0], [math.pi / 2, math.pi / 2, 10.0]).sin()
        np.testing.assert_allclose(R.inf, [0.0, -1.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(R.sup, [1.0, 1.0, 1.0], atol=1e-12)
        S = box([-1.0, 0.0], [2.0, 0.5]).sinh()
        np.testing.assert_allclose(S.inf, np.sinh([-1.0, 0.0]))
        np.testing.assert_allclose(S.sup, np.sinh([2.0, 0.5]))

    def test_sign_and_round(self):
        # sign and round
        R = box([-3.0, 1.0, 0.0, -2.0], [2.0, 3.0, 2.0, 0.0]).sign()
        self.assertEqual(R.inf.tolist(), [-1.0, 1.0, 0.0, -1.0])
        self.assertEqual(R.sup.tolist(), [1.0, 1.0, 1.0, 0.0])
        T = box([-2.5, 0.5, 1.234], [-1.5, 2.5, 2.347]).round()
        self.assertEqual(T.inf.tolist(), [-3.0, 1.0, 1.0])
        self.assertEqual(T.sup.tolist(), [-2.0, 3.0, 2.0])
        np.testing.assert_allclose(box([1.234], [2.347]).round(2).sup, [2.35])
        with self.assertRaises(cora.errors.WrongValue):
            box([1.0], [2.0]).round(-1)
        with self.assertRaises(cora.errors.NotSupported):
            box([1.0], [2.0]).round(1, "significant")

    def test_power(self):
        I = box([-2.0, 1.0, -3.0], [3.0, 2.0, -1.0])
        np.testing.assert_allclose(I.power(2).inf, [0.0, 1.0, 1.0])
        np.testing.assert_allclose((I ** 3).sup, [27.0, 8.0, -1.0])
        np.testing.assert_allclose(box([2.0], [4.0]).power(-1).inf, [0.25])
        np.testing.assert_allclose(box([4.0], [9.0]).power(0.5).sup, [3.0])
        np.testing.assert_allclose(box([2.0, -3.0], [3.0, 2.0]).power(np.array([2.0, 3.0])).inf, [4.0, -27.0])
        E = box([1.0], [3.0])
        np.testing.assert_allclose(cora.Interval.power_of(np.array([2.0]), E).sup, [8.0])
        # powers
        np.testing.assert_allclose((2.0 ** E).sup, [8.0])
        np.testing.assert_allclose(box([2.0], [3.0]).power(box([1.0], [2.0])).sup, [9.0])
        with self.assertRaises(cora.errors.OutOfDomain):
            box([-1.0], [4.0]).power(0.5)
        with self.assertRaises(cora.errors.DimensionMismatch):
            I.power(np.array([1.0, 2.0]))

    def test_rdivide(self):
        D = box([2.0, -4.0, 0.0, -1.0], [4.0, -2.0, 2.0, 2.0])
        R = 1.0 / D
        self.assertEqual(R.inf.tolist(), [0.25, -0.5, 0.5, -INF])
        self.assertEqual(R.sup.tolist(), [0.5, -0.25, INF, INF])
        Q = box([2.0, -4.0, 1.0], [4.0, 6.0, 3.0]).rdivide(np.array([2.0, -2.0, 0.0]))
        self.assertEqual(Q.inf.tolist(), [1.0, -3.0, -INF])
        self.assertEqual(Q.sup.tolist(), [2.0, 2.0, INF])
        P = box([1.0, -2.0], [2.0, 3.0]).rdivide(box([2.0, -4.0], [4.0, -2.0]))
        np.testing.assert_allclose(P.inf, [0.25, -1.5])
        R = cora.Interval.rdivide_of(np.array([1.0, 2.0]), box([2.0, 4.0], [4.0, 8.0]))
        np.testing.assert_allclose(R.sup, [0.5, 0.5])
        with self.assertRaises(cora.errors.OutOfDomain):
            1.0 / box([0.0], [0.0])
        # division by zero and shapes
        with self.assertRaises(cora.errors.SpecialError):
            box([1.0, 2.0], [2.0, 3.0]).rdivide(np.array([1.0, 2.0, 3.0]))

    def test_prod(self):
        v = box([-3.0, 1.0, 0.5], [2.0, 3.0, 5.0])
        self.assertEqual(v.prod().inf.tolist(), [-45.0])
        M = cora.Interval.matrix(np.array([[1.0, -1.0], [2.0, 4.0], [3.0, 1.0]]),
                                 np.array([[2.0, 1.0], [3.0, 4.0], [5.0, 2.0]]))
        np.testing.assert_allclose(M.prod(1).sup, [[30.0, 8.0]])
        np.testing.assert_allclose(M.prod(dim=2).sup, [2.0, 12.0, 10.0])
        with self.assertRaises(cora.errors.WrongValue):
            v.prod(3)

    def test_batches(self):
        I = box([-1.0, 0.5], [2.0, 3.0])
        B = cora.Interval.stack([I, I])
        for name in ("sin", "sinh", "sign", "round"):
            self.assertEqual(getattr(B, name)().inf.shape, (2, 2), name)
        self.assertEqual(B.power(2).sup.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
