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
# PROPERTY: contSet/@capsule/contains_.m
# PROPERTY: contSet/@capsule/convHull_.m
# PROPERTY: contSet/@capsule/enclosePoints.m
# PROPERTY: contSet/@capsule/interval.m
# PROPERTY: contSet/@capsule/isIntersecting_.m
# PROPERTY: contSet/@capsule/mtimes.m

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


def sample(C, rng, count):
    """Points of a capsule as columns (n, count): a point of the axis plus a point of the ball."""
    n = C.c.shape[0]
    t = rng.uniform(-1, 1, count)
    direction = rng.standard_normal((n, count))
    direction /= np.linalg.norm(direction, axis=0)
    return C.c[:, None] + C.generator[:, None] * t + C.r * rng.uniform(0, 1, count) * direction


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

    def test_interval_property(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            C = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            box = C.interval()
            points = sample(C, rng, 20)
            self.assertTrue(np.all(points >= box.inf[:, None] - 1e-12))
            self.assertTrue(np.all(points <= box.sup[:, None] + 1e-12))

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

    # random capsules
    def test_property(self):
        rng = np.random.default_rng(5)
        for _ in range(5):
            C = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            self.assertTrue(C.contains(sample(C, rng, 30)))

    def test_is_intersecting(self):
        C1 = capsule([0, 0], [1, 0], 0.1)
        self.assertTrue(C1.is_intersecting(capsule([0.5, 0], [0, 1], 0.1)))
        self.assertFalse(C1.is_intersecting(capsule([0, 1], [1, 0], 0.4)))
        self.assertTrue(C1.is_intersecting(capsule([0, 1], [1, 0], 0.9)))
        self.assertTrue(C1.is_intersecting(np.array([0.5, 0.05])))
        self.assertFalse(C1.is_intersecting(empty(2)))
        self.assertTrue(cora.is_intersecting(C1, capsule([0.5, 0], [0, 1], 0.1)))

    def test_is_intersecting_property(self):
        rng = np.random.default_rng(7)
        for _ in range(8):
            C1 = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            C2 = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            self.assertEqual(C1.is_intersecting(C2), C2.is_intersecting(C1))
            t = np.linspace(-1, 1, 2001)
            # the distance of the axes
            axis2 = C2.c[:, None] + C2.generator[:, None] * t
            axis1 = C1.c[:, None] + C1.generator[:, None] * t
            distance = np.min(np.linalg.norm(axis2[:, :, None] - axis1[:, None, :], axis=0))
            reach = C1.r + C2.r
            if abs(distance - reach) > 1e-2:
                self.assertEqual(C1.is_intersecting(C2), distance < reach)


class Combination(unittest.TestCase):
    def test_convex_hull(self):
        C1 = capsule([0, 0], [1, 0], 0.5)
        C2 = capsule([4, 0], [0, 1], 0.25)
        H = C1.convex_hull(C2)
        np.testing.assert_allclose(H.c, [2, 0])
        np.testing.assert_allclose(H.generator, [-2, 0])
        self.assertAlmostEqual(H.r, 1.5)
        self.assertTrue(C1.enclose(C2).isequal(H))
        self.assertTrue(cora.convex_hull(C1, C2).isequal(H))
        self.assertTrue(C1.convex_hull(empty(2)).isequal(C1))
        with self.assertRaises(cora.errors.DimensionMismatch):
            C1.convex_hull(cora.Capsule(np.array([1.0])))

    def test_convex_hull_property(self):
        # the convex hull of random capsules
        rng = np.random.default_rng(9)
        for _ in range(5):
            C1 = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            C2 = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            H = C1.convex_hull(C2)
            p1, p2 = sample(C1, rng, 20), sample(C2, rng, 20)
            self.assertTrue(H.contains(p1) and H.contains(p2) and H.contains((p1 + p2) / 2))

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

    def test_linear_map_property(self):
        rng = np.random.default_rng(13)
        for k in range(6):
            C = capsule(rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 1))
            M = (0.3 if k % 2 else 2.0) * rng.standard_normal((2, 3))
            self.assertTrue((M @ C).contains(M @ sample(C, rng, 30)))

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
