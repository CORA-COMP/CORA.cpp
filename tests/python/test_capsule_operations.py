"""test_capsule_operations - the Capsule operations in Python: sums, maps, conversions, samples

CORA authors: Matthias Althoff, Mark Wetzlinger, Victor Gassmann, Niklas Kochdumper
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@capsule/radius.m
# CORA: contSet/@capsule/volume_.m
# CORA: contSet/@capsule/plus.m
# CORA: contSet/@capsule/project.m
# CORA: contSet/@capsule/reduce.m
# CORA: contSet/@capsule/representsa_.m
# CORA: contSet/@capsule/supportFunc_.m
# CORA: contSet/@capsule/randPoint_.m
# CORA: contSet/@capsule/zonotope.m
# CORA: contSet/@capsule/polytope.m
# CORA: contSet/@capsule/ellipsoid.m
# CORA: contSet/@capsule/vertices_.m
# CORA: contSet/@capsule/generateRandom.m
# CORA: contSet/@capsule/dim.m
# CORA: contSet/@capsule/isnan.m
# CORA: contSet/@capsule/plot3D.m
# CORA: contSet/@interval/capsule.m
# CORA: contSet/@zonotope/capsule.m
# PROPERTY: contSet/@capsule/radius.m
# PROPERTY: contSet/@capsule/volume_.m
# PROPERTY: contSet/@capsule/plus.m
# PROPERTY: contSet/@capsule/project.m
# PROPERTY: contSet/@capsule/supportFunc_.m
# PROPERTY: contSet/@capsule/randPoint_.m
# PROPERTY: contSet/@capsule/zonotope.m
# PROPERTY: contSet/@capsule/polytope.m
# PROPERTY: contSet/@capsule/vertices_.m
# PROPERTY: contSet/@capsule/generateRandom.m
# PROPERTY: contSet/@capsule/plot3D.m
# PROPERTY: contSet/@interval/capsule.m
# PROPERTY: contSet/@zonotope/capsule.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def capsule(c, g, r):
    return cora.Capsule(np.array(c, dtype=float), np.array(g, dtype=float), r)


def sample(C, rng, count):
    """Points of a capsule as columns (n, count): a point of the axis plus a point of the ball."""
    n = C.c.shape[0]
    t = rng.uniform(-1, 1, count)
    direction = rng.standard_normal((n, count))
    direction /= np.linalg.norm(direction, axis=0)
    return C.c[:, None] + C.generator[:, None] * t + C.r * rng.uniform(0, 1, count) * direction


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class Operations(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.C = capsule([1.0, 2.0], [3.0, 4.0], 0.5)

    def test_radius_volume_and_small_queries(self):
        self.assertAlmostEqual(self.C.radius(), 5.5)
        self.assertAlmostEqual(self.C.volume(), np.pi * 0.25 + 4 * 0.5 * 5)
        self.assertAlmostEqual(capsule([1.0], [3.0], 0.5).volume(), 7.0)
        self.assertEqual(cora.Capsule.empty(2).volume(), 0.0)
        self.assertFalse(self.C.isnan())
        self.assertEqual(self.C.dim, 2)
        np.testing.assert_array_equal(self.C.reduce(3).c, self.C.c)
        with self.assertRaises(cora.errors.WrongValue):
            self.C.reduce(0)

    def test_plus(self):
        # two capsules: the longer generator stays, the shorter joins the radius
        B = capsule([1.0, 1.0], [1.0, 0.0], 0.5)
        S = self.C + B
        np.testing.assert_array_equal(S.c, [2, 3])
        np.testing.assert_array_equal(S.generator, [3, 4])
        self.assertAlmostEqual(S.r, 2.0)
        self.assertAlmostEqual(self.C.plus(B).r, S.r)
        # a vector, a box and a zonotope
        np.testing.assert_array_equal((self.C + np.array([1.0, 1.0])).c, [2, 3])
        np.testing.assert_array_equal((np.array([1.0, 1.0]) + self.C).c, [2, 3])
        # the sums with a box and a zonotope widen the radius by that of the box
        box = cora.Interval(np.array([0.0, 0.0]), np.array([2.0, 4.0]))
        self.assertAlmostEqual((self.C + box).r, 0.5 + np.sqrt(5.0))
        Z = cora.Zonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 2.0]]))
        self.assertAlmostEqual(self.C.plus(Z).r, 0.5 + np.sqrt(5.0))
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.C.plus(np.array([1.0, 2.0, 3.0]))
        # the sum holds the sums of points of the summands
        rng = np.random.default_rng(3)
        A = capsule([0.0, 1.0, 2.0], [1.0, 0.0, 1.0], 0.4)
        B = capsule([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], 0.6)
        self.assertTrue(A.plus(B).contains(sample(A, rng, 20) + sample(B, rng, 20)))

    def test_project_support_function_and_samples(self):
        C3 = capsule([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.5)
        P = C3.project([2, 0])
        np.testing.assert_array_equal(P.c, [3, 1])
        np.testing.assert_array_equal(P.generator, [6, 4])
        # the support function along two directions
        self.assertAlmostEqual(self.C.support_function(np.array([1.0, 0.0])), 4.5)
        self.assertAlmostEqual(self.C.support_function(np.array([0.0, 2.0])), 13.0)
        # random points: standard ones inside, extreme ones on the boundary
        rng = cora.Rng(5)
        points = self.C.rand_point(30, rng)
        self.assertTrue(self.C.contains(points))
        extreme = self.C.rand_point(10, rng, type="extreme")
        self.assertEqual(extreme.shape, (2, 10))
        self.assertTrue(self.C.contains(extreme))
        with self.assertRaises(cora.errors.NoSpecificAlg):
            self.C.rand_point(3, rng, type="gaussian")

    def test_representsa(self):
        # the types CORA answers without a conversion
        self.assertTrue(cora.Capsule(np.array([0.0, 0.0])).representsa("origin"))
        self.assertTrue(capsule([1.0, 0.0], [0.0, 0.0], 0.0).representsa("point"))
        self.assertFalse(self.C.representsa("point"))
        self.assertTrue(capsule([0.0, 0.0], [1.0, 0.0], 0.0).representsa("zonotope"))
        self.assertTrue(capsule([0.0, 0.0], [0.0, 0.0], 1.0).representsa("ellipsoid"))
        self.assertTrue(self.C.representsa("capsule"))
        self.assertTrue(cora.Capsule.empty(2).representsa("emptySet"))
        with self.assertRaises(cora.errors.NotSupported):
            self.C.representsa("levelSet")
        with self.assertRaises(cora.errors.WrongValue):
            self.C.representsa("sphere")

    def test_conversions(self):
        rng = np.random.default_rng(7)
        # the zonotope holds the capsule, and a ball is the ellipsoid r^2 I
        Z = self.C.zonotope()
        self.assertEqual(Z.generators.shape, (2, 3))
        self.assertTrue(Z.contains(sample(self.C, rng, 10)))
        ball = capsule([1.0, 2.0], [0.0, 0.0], 3.0)
        np.testing.assert_array_equal(ball.ellipsoid().Q, [[9, 0], [0, 9]])
        with self.assertRaises(cora.errors.NoExactAlg):
            self.C.ellipsoid()
        # a segment is a polytope, a proper capsule has an outer one
        segment = capsule([1.0, 1.0], [1.0, 1.0], 0.0)
        self.assertTrue(segment.polytope().contains(np.array([1.5, 1.5])))
        self.assertFalse(segment.polytope().contains(np.array([1.0, 1.5])))
        self.assertTrue(self.C.polytope("outer").contains(sample(self.C, rng, 10)))
        # the capsules of a box and of a zonotope
        box = cora.Interval(np.array([0.0, 0.0, 0.0]), np.array([4.0, 2.0, 1.0]))
        from_box = cora.Capsule(box)
        np.testing.assert_array_equal(from_box.c, [2, 1, 0.5])
        np.testing.assert_array_equal(from_box.generator, [2, 0, 0])
        self.assertAlmostEqual(from_box.r, np.sqrt(1.0 + 0.25))
        zonotope = cora.Zonotope(np.array([1.0, 2.0, 3.0]), rng.standard_normal((3, 4)))
        self.assertTrue(cora.Capsule(zonotope).contains(zonotope))

    def test_vertices_and_random(self):
        # the outline has two half circles of 500 points, closed by the first
        outline = self.C.vertices()[0]
        self.assertEqual(outline.shape, (1001, 2))
        points3 = capsule([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0).vertices_3d()[0]
        self.assertEqual(points3.shape, (400, 3))
        rng = cora.Rng(11)
        R = cora.Capsule.generate_random(rng, dimension=4, center=np.array([1.0, 2.0, 3.0, 4.0]), radius=0.25)
        np.testing.assert_array_equal(R.c, [1, 2, 3, 4])
        self.assertEqual(R.r, 0.25)
        self.assertTrue(1 <= cora.Capsule.generate_random(rng).dim <= 10)


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
