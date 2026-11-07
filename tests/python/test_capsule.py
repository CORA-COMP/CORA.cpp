"""test_capsule - the Capsule class in Python: constructors, queries and set operations.

CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@capsule/capsule.m
# CORA: contSet/@capsule/center.m
# CORA: contSet/@capsule/contains_.m
# CORA: contSet/@capsule/convHull_.m
# CORA: contSet/@capsule/display.m
# CORA: contSet/@capsule/empty.m
# CORA: contSet/@capsule/enclose.m
# CORA: contSet/@capsule/enclosePoints.m
# CORA: contSet/@capsule/interval.m
# CORA: contSet/@capsule/isBounded.m
# CORA: contSet/@capsule/isFullDim.m
# CORA: contSet/@capsule/isIntersecting_.m
# CORA: contSet/@capsule/isequal.m
# CORA: contSet/@capsule/lift_.m
# CORA: contSet/@capsule/mtimes.m
# CORA: contSet/@capsule/origin.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def capsule(c, g, r):
    return cora.Capsule(np.array(c, dtype=float), np.array(g, dtype=float), r)


def empty(n):
    """The empty capsule on the backend of numpy arrays, which the other capsules here are on."""
    with cora.backend("eigen"):
        return cora.Capsule.empty(n)


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class Constructor(unittest.TestCase):
    def test_forms(self):
        point = cora.Capsule(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(point.generator, [0, 0])
        self.assertEqual(point.r, 0.0)
        segment = cora.Capsule(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(segment.generator, [3, 4])
        full = capsule([1, 2], [3, 4], 0.5)
        self.assertEqual(full.dim, 2)
        self.assertEqual(full.r, 0.5)
        np.testing.assert_array_equal(full.c, [1, 2])
        self.assertEqual(full.class_name, "Capsule")
        self.assertFalse(full.is_empty())

    def test_batch(self):
        C = cora.Capsule(np.zeros((3, 2)), np.ones((3, 2)), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(C.c.shape, (3, 2))
        np.testing.assert_allclose(C.r, [0.1, 0.2, 0.3])

    def test_refusals(self):
        with self.assertRaises(cora.errors.WrongValue):
            capsule([1, 2], [3, 4], -0.5)
        # invalid arguments
        with self.assertRaises(cora.errors.WrongValue):
            capsule([1, np.nan], [3, 4], 0.5)
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            capsule([1, 2], [3], 0.5)
        with cora.checks(False):
            self.assertEqual(capsule([1, 2], [3, 4], 0.5).r, 0.5)


class EmptyAndOrigin(unittest.TestCase):
    def test_empty(self):
        E = empty(3)
        self.assertTrue(E.is_empty())
        self.assertEqual(E.dim, 3)
        self.assertEqual(E.r, -1.0)
        self.assertEqual(E.center.shape, (3, 0))
        self.assertTrue(E.interval().inf.shape[-1] == 0 or E.interval().inf.size == 0)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Capsule.empty(-1)

    def test_origin(self):
        O = cora.Capsule.origin(3)
        np.testing.assert_array_equal(O.c, [0, 0, 0])
        self.assertEqual(O.r, 0.0)
        # the origin of dimension 0
        with self.assertRaises(cora.errors.WrongValue):
            cora.Capsule.origin(0)


class Queries(unittest.TestCase):
    def test_center_and_interval(self):
        C = capsule([1, 2], [1, -2], 0.5)
        np.testing.assert_array_equal(C.center, [1, 2])
        box = C.interval()
        np.testing.assert_allclose(box.inf, [-0.5, -0.5])
        np.testing.assert_allclose(box.sup, [2.5, 4.5])

    def test_bounded_and_full_dim(self):
        self.assertTrue(capsule([0, 0], [1, 0], 0.1).is_bounded())
        self.assertTrue(capsule([0, 0], [1, 0], 0.1).is_full_dim())
        # full dimension
        self.assertFalse(capsule([0, 0], [1, 0], 0.0).is_full_dim())
        self.assertTrue(capsule([0], [1], 0.0).is_full_dim())
        self.assertFalse(empty(2).is_full_dim())
        with self.assertRaises(cora.errors.WrongValue):
            capsule([0, 0], [1, 0], 0.1).is_full_dim(tol=-1.0)

    def test_isequal(self):
        C = capsule([1, 2], [3, 4], 0.5)
        self.assertTrue(C.isequal(capsule([1, 2], [3, 4], 0.5)))
        self.assertTrue(C == capsule([1, 2], [3, 4], 0.5))
        self.assertTrue(C != capsule([1, 2], [3, 4], 0.6))
        self.assertTrue(C.isequal(capsule([1, 2.001], [3, 4], 0.5), tol=1e-2))
        self.assertTrue(empty(2).isequal(empty(2)))
        self.assertFalse(empty(2).isequal(C))

    def test_display(self):
        text = repr(capsule([1, 2], [3, 4], 0.5))
        self.assertIn("capsule", text)
        self.assertIn("dimension: 2", text)


class Containment(unittest.TestCase):
    def test_points(self):
        C = capsule([0, 0], [1, 0], 0.5)
        # points in a capsule
        self.assertTrue(C.contains(np.array([0.5, 0.4])))
        self.assertTrue([0.0, 0.0] in C)
        self.assertFalse(C.contains(np.array([0.0, 0.6])))
        self.assertTrue(C.contains(np.array([[0.0, 1.0], [0.0, 0.2]])))
        self.assertFalse(C.contains(np.array([[0.0, 1.0], [0.0, 0.7]])))
        self.assertTrue(C.contains(np.array([0.0, 0.5001]), tol=1e-3))
        with self.assertRaises(cora.errors.DimensionMismatch):
            C.contains(np.array([0.0, 0.0, 0.0]))

    # containment of a capsule, a zonotope, a box and the special sets
    def test_sets(self):
        outer = capsule([0, 0], [2, 0], 1.0)
        self.assertTrue(outer.contains(capsule([0.5, 0], [1, 0], 0.5)))
        self.assertFalse(outer.contains(capsule([0, 0], [2, 0], 1.1)))
        self.assertTrue(outer.contains(cora.Zonotope(np.zeros(2), np.diag([1.0, 0.5]))))
        self.assertTrue(outer.contains(cora.Interval(np.array([-1.0, -0.5]), np.array([1.0, 0.5]))))
        self.assertFalse(outer.contains(cora.Interval(np.array([-1.0, -1.5]), np.array([1.0, 0.5]))))
        self.assertTrue(cora.contains(outer, capsule([0.5, 0], [1, 0], 0.5)))
        self.assertTrue(cora.contains(outer, cora.EmptySet(2)))
        self.assertFalse(cora.contains(outer, cora.Fullspace(2)))
        self.assertTrue(outer.contains(empty(2)))
        self.assertFalse(empty(2).contains(outer))

    def test_is_intersecting(self):
        # capsules near and far, a point and the empty set
        C1 = capsule([0, 0], [1, 0], 0.1)
        self.assertTrue(C1.is_intersecting(capsule([0.5, 0], [0, 1], 0.1)))
        self.assertFalse(C1.is_intersecting(capsule([0, 1], [1, 0], 0.4)))
        self.assertTrue(C1.is_intersecting(capsule([0, 1], [1, 0], 0.9)))
        self.assertTrue(C1.is_intersecting(np.array([0.5, 0.05])))
        self.assertFalse(C1.is_intersecting(empty(2)))
        self.assertTrue(cora.is_intersecting(C1, capsule([0.5, 0], [0, 1], 0.1)))


class Combination(unittest.TestCase):
    def test_convex_hull(self):
        C1 = capsule([0, 0], [1, 0], 0.5)
        C2 = capsule([4, 0], [0, 1], 0.25)
        H = C1.convex_hull(C2)
        np.testing.assert_allclose(H.c, [2, 0])
        np.testing.assert_allclose(H.generator, [-2, 0])
        self.assertAlmostEqual(H.r, 1.5)
        # enclose of two capsules is their hull, also as a function
        self.assertTrue(C1.enclose(C2).isequal(H))
        self.assertTrue(cora.convex_hull(C1, C2).isequal(H))
        self.assertTrue(C1.convex_hull(empty(2)).isequal(C1))
        with self.assertRaises(cora.errors.DimensionMismatch):
            C1.convex_hull(cora.Capsule(np.array([1.0])))

    def test_enclose_points(self):
        line = cora.Capsule.enclose_points(np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
        np.testing.assert_allclose(line.c, [1, 1])
        np.testing.assert_allclose(line.generator, [-1, -1])
        self.assertAlmostEqual(line.r, 0.0)
        self.assertTrue(cora.Capsule.enclose_points(np.zeros((2, 0))).is_empty())
        rng = np.random.default_rng(11)
        points = rng.standard_normal((3, 12))
        C = cora.Capsule.enclose_points(points)
        self.assertTrue(C.contains(points))
        self.assertFalse(cora.Capsule(C.c, C.generator, C.r * 0.999).contains(points))

    def test_linear_map(self):
        C = capsule([1, 2], [1, -1], 0.5)
        # linear maps
        M = np.array([[2.0, 0.0], [0.0, 3.0]])
        for R in (M @ C, M * C, C.linear_map(M)):
            np.testing.assert_allclose(R.c, [2, 6])
            np.testing.assert_allclose(R.generator, [2, -3])
            self.assertAlmostEqual(R.r, 0.5 * np.sqrt(13.0))
        with self.assertRaises(cora.errors.DimensionMismatch):
            C.linear_map(np.ones((1, 3)))

    def test_lift(self):
        C = capsule([1, 2, 3], [4, 5, 6], 0.5)
        L = C.lift(3, [1, 2, 0])
        np.testing.assert_array_equal(L.c, [3, 1, 2])
        np.testing.assert_array_equal(L.generator, [6, 4, 5])
        self.assertEqual(L.r, 0.5)
        with self.assertRaises(cora.errors.NotDefined):
            # lift is not defined
            C.lift(4, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
