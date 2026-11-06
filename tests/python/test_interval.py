"""test_interval - the Interval class in Python: a box, or a matrix of intervals.

CORA authors: Mark Wetzlinger, Matthias Althoff, Sebastian Mair, Tobias Ladner, Niklas Kochdumper,
              Adrian Kulmburg, Dmitry Grebenyuk, Maximilian Perschl
Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
Written:       29-September-2026
Last update:   04-November-2026
Reviewed:      ---
"""
# CORA: contSet/@interval/Inf.m
# CORA: contSet/@interval/abs.m
# CORA: contSet/@interval/acos.m
# CORA: contSet/@interval/acosh.m
# CORA: contSet/@interval/and_.m
# CORA: contSet/@interval/asin.m
# CORA: contSet/@interval/asinh.m
# CORA: contSet/@interval/atan.m
# CORA: contSet/@interval/atan2.m
# CORA: contSet/@interval/atanh.m
# CORA: contSet/@interval/boundaryPoint.m
# CORA: contSet/@interval/cartProd_.m
# CORA: contSet/@interval/cat.m
# CORA: contSet/@interval/center.m
# CORA: contSet/@interval/contains_.m
# CORA: contSet/@interval/convHull_.m
# CORA: contSet/@interval/cos.m
# CORA: contSet/@interval/cosh.m
# CORA: contSet/@interval/ctranspose.m
# CORA: contSet/@interval/diag.m
# CORA: contSet/@interval/dirac.m
# CORA: contSet/@interval/empty.m
# CORA: contSet/@interval/enclose.m
# CORA: contSet/@interval/enclosePoints.m
# CORA: contSet/@interval/enlarge.m
# CORA: contSet/@interval/exp.m
# CORA: contSet/@interval/horzcat.m
# CORA: contSet/@interval/infimum.m
# CORA: contSet/@interval/intervalMatrix.m
# CORA: contSet/@interval/isBounded.m
# CORA: contSet/@interval/generateRandom.m
# CORA: contSet/@interval/interval.m
# CORA: contSet/@interval/mtimes.m
# CORA: contSet/@interval/plus.m
# CORA: contSet/@interval/randPoint_.m
# CORA: contSet/@interval/supportFunc_.m

import unittest

import numpy as np


try:
    import torch
except ImportError:  # numpy-only build: test_numpy.py covers it
    raise unittest.SkipTest("needs torch") from None

import cora


if not cora.has_torch:  # torch is installed, but this build has no torch backend
    raise unittest.SkipTest("needs a build with torch")

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

F64 = dict(dtype=torch.float64)


# ============================================  MAIN  ============================================ #

class Interval(unittest.TestCase):
    def test_bounds_center_and_rad(self):
        I = cora.Interval(torch.tensor([-1.0, 0.0, 2.0], **F64), torch.tensor([3.0, 0.5, 2.0], **F64))
        torch.testing.assert_close(I.inf, torch.tensor([-1.0, 0.0, 2.0], **F64))
        torch.testing.assert_close(I.center, torch.tensor([1.0, 0.25, 2.0], **F64))
        torch.testing.assert_close(I.rad, torch.tensor([2.0, 0.25, 0.0], **F64))
        self.assertEqual(I.dim, 3)

    def test_numpy_boxes_stay_numpy(self):
        I = cora.Interval(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        self.assertIsInstance(I.inf, np.ndarray)
        self.assertIsInstance(I.support_function(np.array([1.0, 1.0])), float)

    def test_support_function_is_the_best_corner(self):
        lo, hi = torch.tensor([-1.0, 0.0], **F64), torch.tensor([2.0, 3.0], **F64)
        I = cora.Interval(lo, hi)
        directions = ([1.0, 1.0], [-1.0, 2.0], [-2.0, -1.0])
        for d in (torch.tensor(d, **F64) for d in directions):
            self.assertAlmostEqual(float(I.support_function(d)), float(torch.maximum(d * lo, d * hi).sum()))

    def test_contains(self):
        I = cora.Interval(torch.tensor([0.0, -1.0], **F64), torch.tensor([2.0, 1.0], **F64))
        self.assertTrue(I.contains(torch.tensor([1.0, 0.0], **F64)))
        self.assertTrue(I.contains(torch.tensor([0.0, 1.0], **F64)))  # a corner belongs to the box
        self.assertFalse(I.contains(torch.tensor([2.1, 0.0], **F64)))

    def test_mtimes_is_the_hull_of_the_image(self):
        I = cora.Interval(torch.tensor([-1.0, 0.0, 1.0], **F64), torch.tensor([1.0, 2.0, 3.0], **F64))
        M = torch.randn(3, 3, **F64)
        image = I.linear_map(M)
        for _ in range(50):
            x = I.inf + torch.rand(3, **F64) * (I.sup - I.inf)
            self.assertTrue(image.contains(M @ x))

    def test_plus_adds_the_bounds(self):
        S = cora.Interval(torch.tensor([0.0, 1.0], **F64), torch.tensor([1.0, 3.0], **F64)).plus(
            cora.Interval(torch.tensor([-2.0, 0.5], **F64), torch.tensor([0.0, 0.5], **F64)))
        torch.testing.assert_close(S.inf, torch.tensor([-2.0, 1.5], **F64))
        torch.testing.assert_close(S.sup, torch.tensor([1.0, 3.5], **F64))

    def test_rand_point_fills_the_box(self):
        I = cora.Interval(torch.tensor([0.0, -1.0], **F64), torch.tensor([2.0, 1.0], **F64))
        P = I.rand_point(2000, cora.Rng(4))
        self.assertEqual(tuple(P.shape), (2, 2000))
        self.assertTrue(bool((P.T >= I.inf).all() and (P.T <= I.sup).all()))
        self.assertLess(float((P.mean(-1) - I.center).abs().max()), 0.1)

    def test_generate_random(self):
        # random intervals are ordered
        I = cora.Interval.generate_random(cora.Rng(2), 4)
        self.assertEqual(I.dim, 4)
        self.assertTrue(bool((I.inf <= I.sup).all()))

    def test_generate_random_takes_the_options_of_cora(self):
        rng = cora.Rng(2)
        I = cora.Interval.generate_random(rng, center=[1.0, -1.0, 0.5], max_radius=[2.0, 0.0, 4.0])
        self.assertEqual(I.dim, 3)
        self.assertTrue(np.allclose(np.asarray(I.center), [1.0, -1.0, 0.5]))
        self.assertTrue(bool((np.asarray(I.rad) <= [1.0, 0.0, 2.0]).all()))
        S = cora.Interval.generate_random(rng, 4, max_radius=6.0)
        self.assertTrue(bool((np.asarray(S.rad) <= 3.0).all()))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval.generate_random(rng, 2, center=[1.0, 2.0, 3.0])

    def test_an_interval_can_be_a_matrix(self):
        # The correction matrix of a system is an Interval whose bounds are matrices.
        sys = cora.LinearSys(torch.tensor([[-0.2, 1.0], [-1.0, -0.2]], **F64))
        F = sys.correction_matrix_state(0.1, 8)
        self.assertEqual(tuple(F.inf.shape), (2, 2))
        self.assertTrue(bool((F.inf <= F.sup).all()))
        # MATLAB CORA's taylorMatrices gives F.sup(1, 1) = 0.0012000413732381331 for this system.
        self.assertAlmostEqual(float(F.sup[0, 0]), 0.0012000413732381331, places=14)


def box(lo, hi):
    return cora.Interval(torch.tensor(lo, **F64), torch.tensor(hi, **F64))


class IntervalMethods(unittest.TestCase):
    def test_center_of_unbounded_and_empty_boxes(self):
        inf = float("inf")
        self.assertTrue(bool(torch.isnan(box([0.0, -inf], [inf, 2.0]).center).all()))
        self.assertEqual(box([-inf], [inf]).center.tolist(), [0.0])
        self.assertEqual(cora.Interval.unbounded(2).center.tolist(), [0.0, 0.0])

    def test_unbounded(self):
        I = cora.Interval.unbounded(3)
        self.assertEqual(I.inf.tolist(), [-float("inf")] * 3)
        self.assertEqual(I.sup.tolist(), [float("inf")] * 3)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval.unbounded(-1)

    def test_abs(self):
        A = box([-3.0, -2.0, 1.0], [-1.0, 4.0, 2.0]).abs()
        self.assertEqual(A.inf.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(abs(box([-3.0], [-1.0])).sup.tolist(), [3.0])

    def test_intersect(self):
        # intersection
        I, J = box([1.0, -1.0], [2.0, 1.0]), box([1.5, -2.0], [2.5, 0.0])
        self.assertEqual((I & J).inf.tolist(), [1.5, -1.0])
        self.assertEqual(I.intersect(J).sup.tolist(), [2.0, 0.0])
        empty = I.intersect(box([5.0, 5.0], [6.0, 6.0]))
        self.assertEqual(tuple(empty.inf.shape), (2, 0))
        with self.assertRaises(cora.errors.DimensionMismatch):
            I.intersect(box([0.0], [1.0]))

    def test_cart_prod_and_cat(self):
        I, J = box([-1.0, -3.0], [1.0, 6.0]), box([-2.0], [4.0])
        self.assertEqual(I.cart_prod(J).inf.tolist(), [-1.0, -3.0, -2.0])
        self.assertEqual(cora.cart_prod(I, J).sup.tolist(), [1.0, 6.0, 4.0])
        self.assertEqual(I.cart_prod(torch.tensor([7.0], **F64)).sup.tolist(), [1.0, 6.0, 7.0])
        self.assertEqual(cora.Interval.cat(1, [I, J]).inf.tolist(), [-1.0, -3.0, -2.0])
        joined = cora.Interval.cat(2, [I, I])
        self.assertEqual(tuple(joined.inf.shape), (2, 2))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval.cat(0, [I])
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.Interval.cat(2, [I, J])

    def test_boundary_point(self):
        # boundary points
        I = box([1.0, -1.0], [4.0, 3.0])
        torch.testing.assert_close(I.boundary_point(torch.tensor([1.0, 1.0], **F64)),
                                   torch.tensor([4.0, 2.5], **F64))
        x = I.boundary_point(torch.tensor([-2.0, -1.0], **F64), start=torch.tensor([3.0, 2.0], **F64))
        torch.testing.assert_close(x, torch.tensor([1.0, 1.0], **F64))
        with self.assertRaises(cora.errors.WrongValue):
            I.boundary_point(torch.zeros(2, **F64))

    def test_inverse_functions(self):
        cases = {"acos": ([0.0, -1.0], [0.5, 1.0]), "asin": ([0.0, -1.0], [0.5, 1.0]),
                 "atan": ([-1.0, 0.0], [1.0, 3.0]), "acosh": ([1.0, 2.0], [2.0, 10.0]),
                 "asinh": ([-1.0, 0.0], [1.0, 2.0]), "atanh": ([-0.5, 0.0], [0.5, 0.5])}
        for name, (lo, hi) in cases.items():
            R = getattr(box(lo, hi), name)()
            f = getattr(torch, name)
            values = torch.stack([f(torch.tensor(lo, **F64)), f(torch.tensor(hi, **F64))])
            torch.testing.assert_close(R.inf, values.min(0).values, msg=name)
            torch.testing.assert_close(R.sup, values.max(0).values, msg=name)
        for name, bad in (("acos", ([-1.5], [0.0])), ("asin", ([0.0], [1.5])),
                          ("atanh", ([0.0], [1.5])), ("acosh", ([0.5], [2.0]))):
            with self.assertRaises(cora.errors.OutOfDomain, msg=name):
                # out of domain
                getattr(box(*bad), name)()

    def test_atan2_is_the_hull_of_the_corners(self):
        Y, X = box([1.0], [2.0]), box([1.0], [2.0])
        R = Y.atan2(X)
        self.assertAlmostEqual(float(R.inf[0]), 0.4636476090008061, places=14)
        self.assertAlmostEqual(float(R.sup[0]), 1.1071487177940904, places=14)
        cut = box([-1.0], [1.0]).atan2(box([-2.0], [-1.0]))  # across the branch cut
        self.assertGreaterEqual(float(cut.sup[0]), 3.141592653589793)
        self.assertLessEqual(float(cut.inf[0]), -3.141592653589793)

    def test_exp_cos_cosh(self):
        torch.testing.assert_close(box([-1.0], [2.0]).exp().sup, torch.exp(torch.tensor([2.0], **F64)))
        torch.testing.assert_close(box([-1.0], [2.0]).exp().inf, torch.exp(torch.tensor([-1.0], **F64)))
        C = box([3.0, 0.0, -10.0], [4.0, 1.0, 20.0]).cos()  # [3, 4] holds pi
        torch.testing.assert_close(C.inf, torch.tensor([-1.0, np.cos(1.0), -1.0], **F64))
        torch.testing.assert_close(C.sup, torch.tensor([np.cos(4.0), 1.0, 1.0], **F64))
        H = box([-2.0, 1.0, -1.0], [-1.0, 2.0, 3.0]).cosh()  # left of 0, right of 0, around 0
        torch.testing.assert_close(H.inf, torch.tensor([np.cosh(1.0), np.cosh(1.0), 1.0], **F64))
        torch.testing.assert_close(H.sup, torch.tensor([np.cosh(2.0), np.cosh(2.0), np.cosh(3.0)], **F64))

    def test_convex_hull_and_enclose(self):
        I, J = box([-1.0, 0.0], [1.0, 2.0]), box([0.0, 3.0], [4.0, 5.0])
        self.assertEqual(I.convex_hull(J).inf.tolist(), [-1.0, 0.0])
        self.assertEqual(I.enclose(J).sup.tolist(), [4.0, 5.0])
        self.assertEqual(cora.convex_hull(I, J).sup.tolist(), [4.0, 5.0])
        self.assertEqual(I.convex_hull(cora.Interval.empty(2)).sup.tolist(), [1.0, 2.0])
        for f in (I.convex_hull, I.enclose):
            with self.assertRaises(cora.errors.DimensionMismatch):
                f(box([0.0], [1.0]))

    def test_enclose_points(self):
        P = torch.tensor([[1.0, -2.0, 3.0], [0.0, 5.0, -1.0]], **F64)
        I = cora.Interval.enclose_points(P)
        self.assertEqual(I.inf.tolist(), [-2.0, -1.0])
        self.assertEqual(I.sup.tolist(), [3.0, 5.0])
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval.enclose_points(torch.zeros(2, 0, **F64))

    def test_empty_and_is_bounded(self):
        E = cora.Interval.empty(3)
        # empty and unbounded boxes
        self.assertEqual(tuple(E.inf.shape), (3, 0))
        self.assertTrue(E.is_bounded())
        self.assertTrue(box([0.0], [1.0]).is_bounded())
        self.assertFalse(cora.Interval.unbounded(2).is_bounded())
        self.assertFalse(box([0.0], [float("inf")]).is_bounded())
        with self.assertRaises(cora.errors.WrongValue):
            cora.Interval.empty(-1)

    def test_enlarge(self):
        I = box([0.0, 1.0], [2.0, 5.0])
        self.assertEqual(I.enlarge(2.0).inf.tolist(), [-1.0, -1.0])
        self.assertEqual(I.enlarge(torch.tensor([0.5, 0.0], **F64)).sup.tolist(), [1.5, 3.0])
        self.assertEqual(cora.Interval.unbounded(1).enlarge(2.0).sup.tolist(), [float("inf")])
        with self.assertRaises(cora.errors.WrongValue):
            I.enlarge(-1.0)
        with self.assertRaises(cora.errors.NotSupported):
            cora.Interval.unbounded(1).enlarge(0.5)

    def test_matrix_methods(self):
        a, b = box([1.0, 2.0], [3.0, 4.0]), box([5.0, 6.0], [7.0, 8.0])
        M = cora.Interval.horzcat([a, b])
        self.assertEqual(tuple(M.inf.shape), (2, 2))
        # matrices of intervals
        self.assertEqual(M.inf.tolist(), [[1.0, 5.0], [2.0, 6.0]])
        self.assertEqual(M.ctranspose().sup.tolist(), [[3.0, 4.0], [7.0, 8.0]])
        self.assertEqual(cora.Interval.horzcat([a, torch.tensor([9.0, 9.5], **F64)]).inf.tolist(),
                         [[1.0, 9.0], [2.0, 9.5]])
        self.assertEqual(a.diag().inf.tolist(), [[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(a.diag(1).sup.tolist(), [[0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
        self.assertEqual(M.diag().inf.tolist(), [1.0, 6.0])
        self.assertEqual(M.infimum().tolist(), [[1.0, 5.0], [2.0, 6.0]])
        self.assertEqual(M.interval_matrix().sup.tolist(), M.sup.tolist())
        with self.assertRaises(cora.errors.NotSupported):
            M.diag(2)
        with self.assertRaises(cora.errors.DimensionMismatch):
            cora.Interval.horzcat([a, box([0.0], [1.0])])

    def test_dirac(self):
        inf = float("inf")
        I = box([-3.0, -1.0, 0.0, -2.0], [-1.0, 1.0, 3.0, 0.0])
        self.assertEqual(I.dirac().sup.tolist(), [0.0, inf, inf, inf])
        self.assertEqual(I.dirac(1).inf.tolist(), [0.0, -inf, -inf, 0.0])
        self.assertEqual(I.dirac(2).inf.tolist(), [0.0, -inf, -inf, -inf])
        with self.assertRaises(cora.errors.WrongValue):
            # batches
            I.dirac(-1)

    def test_batches(self):
        I = cora.Interval(torch.tensor([[-1.0, 0.5], [0.0, 2.0]], **F64),
                          torch.tensor([[1.0, 1.0], [3.0, 4.0]], **F64))
        self.assertEqual(tuple(I.abs().sup.shape), (2, 2))
        self.assertEqual(tuple(I.cart_prod(I).inf.shape), (2, 4))
        self.assertEqual(tuple(I.atan().inf.shape), (2, 2))
        self.assertEqual(tuple(I.center.shape), (2, 2))


class IntervalGradients(unittest.TestCase):
    """Autograd through the operations equals central differences (piecewise ones off their kinks)."""

    def check(self, f, lo, hi):
        lo = torch.tensor(lo, requires_grad=True, **F64)
        hi = torch.tensor(hi, requires_grad=True, **F64)

        def bounds(a, b):
            R = f(cora.Interval(a, b))
            return torch.stack([R.inf, R.sup])

        self.assertTrue(torch.autograd.gradcheck(bounds, (lo, hi), atol=1e-5, rtol=1e-4))

    def test_gradients(self):
        lo, hi = [0.1, -0.6, 0.3], [0.7, 0.4, 0.9]
        for name in ("acos", "asin", "atan", "asinh", "atanh", "abs", "exp", "cos", "cosh"):
            with self.subTest(name):
                self.check(lambda I, name=name: getattr(I, name)(), lo, hi)
        self.check(lambda I: I.acosh(), [1.2, 2.0], [3.0, 5.0])
        self.check(lambda I: I.atan2(I.abs() + torch.ones(3, **F64)), lo, hi)
        self.check(lambda I: I.intersect(I + torch.tensor([0.1, 0.1, 0.1], **F64)), lo, hi)
        self.check(lambda I: I.cart_prod(I), lo, hi)
        self.check(lambda I: I.convex_hull(I + torch.full((3,), 0.05, **F64)), lo, hi)
        self.check(lambda I: I.enlarge(1.5), lo, hi)
        self.check(lambda I: I.interval_matrix(), lo, hi)
        self.check(lambda I: cora.Interval.horzcat([I, I]).ctranspose(), lo, hi)
        # gradients of structural operations
        self.check(lambda I: I.diag(), lo, hi)
        self.check(lambda I: cora.Interval.cat(1, [I, I]), lo, hi)
        self.check(lambda I: cora.Interval(I.center, I.center + 1.0), lo, hi)
        # off the kink where two faces are reached at once
        direction = torch.tensor([1.0, 0.5, -1.0], **F64)
        self.check(lambda I: cora.Interval(I.boundary_point(direction),
                                           I.boundary_point(direction) + 1.0),
                   [0.1, -0.6, 0.2], hi)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
