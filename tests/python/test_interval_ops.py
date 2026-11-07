"""test_interval_ops - the comparisons, differences, quotients, norms and embeddings of an interval.

CORA authors: Niklas Kochdumper, Mark Wetzlinger, Matthias Althoff, Victor Gassmann,
              Tobias Ladner, Dmitry Grebenyuk
Authors:       Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@interval/isFullDim.m
# CORA: contSet/@interval/isIntersecting_.m
# CORA: contSet/@interval/isequal.m
# CORA: contSet/@interval/kron.m
# CORA: contSet/@interval/le.m
# CORA: contSet/@interval/lift_.m
# CORA: contSet/@interval/log.m
# CORA: contSet/@interval/lt.m
# CORA: contSet/@interval/max.m
# CORA: contSet/@interval/min.m
# CORA: contSet/@interval/minkDiff.m
# CORA: contSet/@interval/minus.m
# CORA: contSet/@interval/mpower.m
# CORA: contSet/@interval/mrdivide.m
# CORA: contSet/@interval/norm_.m
# CORA: contSet/@interval/or.m
# CORA: contSet/@interval/origin.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

INF = float("inf")


def box(lo, hi):
    return cora.Interval(np.array(lo), np.array(hi))


# ============================================  MAIN  ============================================ #

class Predicates(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_is_full_dim(self):
        self.assertTrue(box([0.0, 1.0], [1.0, 3.0]).is_full_dim())
        self.assertFalse(box([0.0, 1.0], [1.0, 1.0]).is_full_dim())
        with cora.tolerance(1e-8):  # CORA's withinTol default
            self.assertFalse(box([0.0, 1.0], [1.0, 1.0 + 1e-9]).is_full_dim())
        self.assertFalse(cora.Interval.empty(2).is_full_dim())

    def test_is_intersecting(self):
        I = box([0.0, 0.0], [2.0, 2.0])
        self.assertTrue(I.is_intersecting(box([1.0, 1.0], [3.0, 3.0])))
        self.assertTrue(I.is_intersecting(box([2.0, 2.0], [3.0, 3.0])))
        apart = box([2.5, 0.0], [3.0, 1.0])
        self.assertFalse(I.is_intersecting(apart))
        self.assertTrue(I.is_intersecting(apart, tol=0.6))
        self.assertFalse(I.is_intersecting(apart, tol=0.2))
        self.assertTrue(I.is_intersecting(np.array([1.0, 2.0])))
        self.assertFalse(I.is_intersecting(np.array([1.0, 2.5])))
        self.assertTrue(cora.is_intersecting(I, box([1.0, 1.0], [3.0, 3.0])))
        with self.assertRaises(cora.errors.DimensionMismatch):
            I.is_intersecting(box([0.0], [1.0]))
        with self.assertRaises(cora.errors.WrongValue):
            I.is_intersecting(I, tol=-1.0)

    def test_isequal(self):
        I = box([0.0, -1.0], [1.0, 2.0])
        self.assertTrue(I.isequal(I))
        # equality up to tol
        self.assertFalse(I.isequal(box([0.0, -1.0], [1.0, 2.5])))
        self.assertFalse(I.isequal(box([0.0, -1.0], [1.0 + 1e-9, 2.0])))
        self.assertTrue(I.isequal(box([0.0, -1.0], [1.0 + 1e-9, 2.0]), tol=1e-8))
        self.assertFalse(I.isequal(box([0.0], [1.0])))
        self.assertTrue(box([1.0, 2.0], [1.0, 2.0]).isequal(np.array([[1.0], [2.0]])))
        H = box([0.0, -INF], [INF, 1.0])
        self.assertTrue(H.isequal(H))
        self.assertFalse(H.isequal(box([0.0, -INF], [1e300, 1.0])))
        with self.assertRaises(cora.errors.WrongValue):
            I.isequal(I, tol=-1.0)

    def test_comparisons_are_containment(self):
        small, big = box([0.0, 1.0], [1.0, 2.0]), box([-1.0, 0.0], [3.0, 3.0])
        self.assertTrue(small.less_equal(big) and small <= big)
        self.assertTrue(small.less(big) and small < big)
        self.assertFalse(big <= small)
        self.assertTrue(small <= small)
        self.assertFalse(small < small)
        with self.assertRaises(cora.errors.DimensionMismatch):
            _ = small <= box([0.0], [1.0])
        with self.assertRaises(TypeError):
            # ordering is refused
            _ = small <= 1.0


class Arithmetic(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_minus(self):
        I, J = box([0.0, 1.0], [2.0, 4.0]), box([-1.0, 0.0], [1.0, 3.0])
        for R in (I.minus(J), I - J):
            np.testing.assert_allclose(R.inf, [-1.0, -2.0])
            np.testing.assert_allclose(R.sup, [3.0, 4.0])
        np.testing.assert_allclose((I - np.array([1.0, 1.0])).inf, [-1.0, 0.0])
        np.testing.assert_allclose(I.minus(1.0).sup, [1.0, 3.0])
        R = np.array([1.0, 1.0]) - I
        np.testing.assert_allclose(R.inf, [-1.0, -3.0])
        np.testing.assert_allclose(R.sup, [1.0, 0.0])
        with self.assertRaises(cora.errors.DimensionMismatch):
            I.minus(box([0.0], [1.0]))

    def test_divide(self):
        I = box([2.0, -4.0], [4.0, 6.0])
        np.testing.assert_allclose((I / 2.0).sup, [2.0, 3.0])
        R = I.divide(-2.0)
        # division
        np.testing.assert_allclose(R.inf, [-2.0, -3.0])
        np.testing.assert_allclose(R.sup, [-1.0, 2.0])
        self.assertEqual((I / 0.0).inf.tolist(), [-INF, -INF])
        with self.assertRaises(cora.errors.OutOfDomain):
            box([1.0], [INF]).divide(INF)

    def test_max_min_and_unite(self):
        I, J = box([0.0, -1.0], [2.0, 1.0]), box([1.0, -2.0], [3.0, 0.0])
        np.testing.assert_allclose(I.max(J).inf, [1.0, -1.0])
        np.testing.assert_allclose(I.max(J).sup, [3.0, 1.0])
        np.testing.assert_allclose(I.min(J).inf, [0.0, -2.0])
        np.testing.assert_allclose(I.min(J).sup, [2.0, 0.0])
        np.testing.assert_allclose(I.max(np.array([1.0, 0.5])).inf, [1.0, 0.5])
        np.testing.assert_allclose(I.min(0.5).sup, [0.5, 0.5])
        Z = cora.Zonotope(np.array([3.0, 3.0]), np.eye(2))
        np.testing.assert_allclose(I.max(Z).inf, [2.0, 2.0])
        np.testing.assert_allclose(I.max(), I.sup)
        np.testing.assert_allclose(I.min(), I.inf)
        np.testing.assert_allclose(I.unite(J).inf, [0.0, -2.0])
        np.testing.assert_allclose(I.unite(J).sup, [3.0, 1.0])
        np.testing.assert_allclose(I.unite(np.array([-1.0, 3.0])).sup, [2.0, 3.0])
        # union, maximum and minimum
        np.testing.assert_allclose(cora.unite(I, J).sup, [3.0, 1.0])
        for f in (I.max, I.min, I.unite):
            with self.assertRaises(cora.errors.DimensionMismatch):
                f(box([0.0], [1.0]))

    def test_minkowski_difference(self):
        I = box([0.0, 0.0], [10.0, 4.0])
        R = I.minkowski_difference(box([1.0, -1.0], [3.0, 1.0]))
        np.testing.assert_allclose(R.inf, [-1.0, 1.0])
        np.testing.assert_allclose(R.sup, [7.0, 3.0])
        np.testing.assert_allclose(I.minkowski_difference(np.array([1.0, 2.0])).sup, [9.0, 2.0])
        Z = cora.Zonotope(np.zeros(2), 0.5 * np.eye(2))
        B = box([0.0, 0.0], [4.0, 4.0])
        np.testing.assert_allclose(B.minkowski_difference(Z).inf, [0.5, 0.5])
        np.testing.assert_allclose(B.minkowski_difference(Z, "inner").sup, [3.5, 3.5])
        self.assertTrue(box([0.0], [1.0]).minkowski_difference(box([-1.0], [1.0])).inf.size == 0)
        with self.assertRaises(cora.errors.NoExactAlg):
            I.minkowski_difference(Z, "exact")
        with self.assertRaises(cora.errors.WrongValue):
            I.minkowski_difference(Z, "outer")

    def test_kron_and_matrix_power(self):
        # Kronecker products
        I = cora.Interval.matrix(np.array([[1.0, -1.0]]), np.array([[2.0, 1.0]]))
        J = cora.Interval.matrix(np.array([[0.0], [2.0]]), np.array([[1.0], [3.0]]))
        P = I.kron(J)
        np.testing.assert_allclose(P.inf, [[0.0, -1.0], [2.0, -3.0]])
        np.testing.assert_allclose(P.sup, [[2.0, 1.0], [6.0, 3.0]])
        Q = J.kron(np.array([[1.0, -1.0]]), left=True)
        np.testing.assert_allclose(Q.inf, [[0.0, -1.0], [2.0, -3.0]])
        np.testing.assert_allclose(Q.sup, [[1.0, 0.0], [3.0, -2.0]])
        A = cora.Interval.matrix(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[2.0, 1.0], [0.0, 1.0]]))
        np.testing.assert_allclose(A.matrix_power(0).inf, np.eye(2))
        np.testing.assert_allclose(A.matrix_power(2).sup, [[4.0, 3.0], [0.0, 1.0]])
        np.testing.assert_allclose(A.matrix_power(3).sup, [[8.0, 7.0], [0.0, 1.0]])
        rng = np.random.default_rng(4)
        M = rng.normal(size=(3, 3))
        P4 = cora.Interval.matrix(M, M).matrix_power(4)
        np.testing.assert_allclose(P4.sup, np.linalg.matrix_power(M, 4), rtol=1e-9)
        with self.assertRaises(cora.errors.WrongValue):
            A.matrix_power(-1)
        with self.assertRaises(cora.errors.NotSupported):
            cora.Interval.matrix(np.array([[1.0]]), np.array([[2.0]])).matrix_power(2)

    # logarithm
    def test_log(self):
        R = box([1.0, 0.5], [np.e, 8.0]).log()
        np.testing.assert_allclose(R.inf, [0.0, np.log(0.5)])
        np.testing.assert_allclose(R.sup, [1.0, np.log(8.0)])
        self.assertEqual(box([0.0], [1.0]).log().inf.tolist(), [-INF])
        rng = np.random.default_rng(4)
        lo = rng.uniform(0.1, 3.0, 4)
        I = box(lo, lo + rng.uniform(0.0, 2.0, 4))
        R = I.log()
        for _ in range(20):
            x = I.inf + rng.uniform(size=4) * (I.sup - I.inf)
            self.assertTrue(bool(np.all(R.inf <= np.log(x)) and np.all(np.log(x) <= R.sup)))
        with self.assertRaises(cora.errors.OutOfDomain):
            box([-1.0], [1.0]).log()


class Embedding(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_norm(self):
        I = box([-3.0, -4.0], [1.0, 2.0])
        self.assertAlmostEqual(I.norm(), 5.0)
        # norms
        self.assertAlmostEqual(I.norm(1), 7.0)
        self.assertAlmostEqual(I.norm(INF), 4.0)
        self.assertAlmostEqual(I.norm("fro"), 5.0)
        self.assertEqual(cora.Interval.empty(2).norm(), -INF)
        with self.assertRaises(cora.errors.WrongValue):
            I.norm(3)

    def test_lift_and_origin(self):
        I = box([1.0, 3.0], [2.0, 4.0])
        R = I.lift(4, [2, 0])
        self.assertEqual(R.inf.tolist(), [3.0, -INF, 1.0, -INF])
        self.assertEqual(R.sup.tolist(), [4.0, INF, 2.0, INF])
        with self.assertRaises(cora.errors.WrongValue):
            I.lift(3, [0, 3])
        with self.assertRaises(cora.errors.NotSupported):
            cora.Interval.empty(2).lift(3, [0, 1])
        O = cora.Interval.origin(3)
        self.assertEqual(O.inf.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(O.sup.tolist(), [0.0, 0.0, 0.0])
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval.origin(0)


# gradients
class Gradients(unittest.TestCase):
    """Autograd through the operations equals central differences (piecewise ones off their kinks)."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("needs torch")
        if not cora.has_torch:
            self.skipTest("needs a build with torch")

    def check(self, f, lo, hi):
        import torch
        lo = torch.tensor(lo, requires_grad=True, dtype=torch.float64)
        hi = torch.tensor(hi, requires_grad=True, dtype=torch.float64)

        def bounds(a, b):
            R = f(cora.Interval.matrix(a, b) if lo.dim() == 2 else cora.Interval(a, b))
            return torch.stack([R.inf, R.sup])

        self.assertTrue(torch.autograd.gradcheck(bounds, (lo, hi), atol=1e-5, rtol=1e-4))

    def test_gradients(self):
        import torch
        lo, hi = [0.1, -0.6, 0.3], [0.7, 0.4, 0.9]
        shift = torch.tensor([0.2, 0.1, -0.1], dtype=torch.float64)
        self.check(lambda I: I.log(), [0.5, 1.5, 2.0], [1.2, 2.5, 3.0])
        self.check(lambda I: I - shift, lo, hi)
        self.check(lambda I: shift - I, lo, hi)
        self.check(lambda I: I.minus(I + shift * 3.0), lo, hi)
        # gradients of binary operations
        self.check(lambda I: I.divide(-2.5), lo, hi)
        self.check(lambda I: I.max(I + shift), lo, hi)
        self.check(lambda I: I.min(I + shift), lo, hi)
        self.check(lambda I: I.unite(I + shift * 4.0), lo, hi)
        self.check(lambda I: I.minkowski_difference(I.enlarge(0.5)), lo, hi)
        # positive entries, away from the sign changes of the corner products
        low, high = [[1.0, 0.5], [0.3, 1.2]], [[1.5, 0.9], [0.8, 1.6]]
        self.check(lambda A: A.matrix_power(3), low, high)
        self.check(lambda A: A.kron(A), low, high)
        self.check(lambda A: A.kron(torch.tensor([[1.0, -2.0]], dtype=torch.float64)), low, high)
        a = torch.tensor(lo, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(hi, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x, y: cora.Interval(x, y).norm().reshape(1), (a, b)))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
