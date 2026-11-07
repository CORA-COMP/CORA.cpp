"""test_ellipsoid_operations - the Ellipsoid operations in Python: maps, sums, conversions, samples

CORA authors: Victor Gassmann, Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@ellipsoid/supportFunc_.m
# CORA: contSet/@ellipsoid/mtimes.m
# CORA: contSet/@ellipsoid/plus.m
# CORA: contSet/@ellipsoid/project.m
# CORA: contSet/@ellipsoid/lift_.m
# CORA: contSet/@ellipsoid/isIntersecting_.m
# CORA: contSet/@ellipsoid/isequal.m
# CORA: contSet/@ellipsoid/minkDiff.m
# CORA: contSet/@ellipsoid/norm_.m
# CORA: contSet/@ellipsoid/origin.m
# CORA: contSet/@ellipsoid/radius.m
# CORA: contSet/@ellipsoid/reduce.m
# CORA: contSet/@ellipsoid/vertices_.m
# CORA: contSet/@ellipsoid/volume_.m
# CORA: contSet/@ellipsoid/convHull_.m
# CORA: contSet/@ellipsoid/or.m
# CORA: contSet/@ellipsoid/zonotope.m
# CORA: contSet/@ellipsoid/distance.m
# CORA: contSet/@ellipsoid/dim.m
# CORA: contSet/@ellipsoid/generateRandom.m
# CORA: contSet/@ellipsoid/randPoint_.m
# CORA: contSet/@ellipsoid/isnan.m
# CORA: contSet/@ellipsoid/plot3D.m
# CORA: contSet/@interval/ellipsoid.m
# CORA: contSet/@zonotope/ellipsoid.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def ellipsoid(Q, q=None, tol=None):
    return cora.Ellipsoid(np.array(Q, dtype=float), None if q is None else np.array(q, dtype=float), tol)


def random_ellipsoid(rng, n):
    """Q = A A' + I/2 and a normal center."""
    A = rng.standard_normal((n, n))
    Q = A @ A.T + 0.5 * np.eye(n)
    return cora.Ellipsoid(0.5 * (Q + Q.T), rng.standard_normal(n))


def boundary_points(E, rng, count):
    """Points q + G u of the boundary."""
    G = E.generators()
    U = rng.standard_normal((G.shape[1], count))
    return E.q[:, None] + G @ (U / np.linalg.norm(U, axis=0))


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class Operations(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.E = ellipsoid([[4.0, 0.0], [0.0, 1.0]], [1.0, 2.0])

    def test_support_function_radius_volume_norm(self):
        # the support function along the axes and a diagonal
        self.assertAlmostEqual(self.E.support_function(np.array([1.0, 0.0])), 3.0)
        self.assertAlmostEqual(self.E.support_function(np.array([1.0, 1.0])), 3.0 + np.sqrt(5.0))
        np.testing.assert_allclose(self.E.radius(2), [2.0, 1.0])
        self.assertAlmostEqual(self.E.volume(), 2 * np.pi)
        self.assertAlmostEqual(ellipsoid([[4.0, 0.0], [0.0, 1.0]]).norm(), 2.0)
        with self.assertRaises(cora.errors.NotSupported):
            self.E.norm()
        self.assertFalse(self.E.isnan())
        self.assertEqual(self.E.dim, 2)
        np.testing.assert_array_equal(self.E.reduce(2).Q, self.E.Q)
        np.testing.assert_array_equal(cora.Ellipsoid.origin(3).Q, np.zeros((3, 3)))
        with self.assertRaises(cora.errors.WrongValue):
            self.E.radius(3)

    def test_maps(self):
        # a row map, a projection and a permutation
        R = self.E.linear_map(np.array([[1.0, 2.0]]))
        np.testing.assert_allclose(R.Q, [[8.0]])
        np.testing.assert_allclose(R.q, [5.0])
        np.testing.assert_allclose((np.array([[1.0, 2.0]]) @ self.E).Q, [[8.0]])
        S = self.E.project([1, 0])
        np.testing.assert_allclose(S.Q, [[1, 0], [0, 4]])
        np.testing.assert_array_equal(S.q, [2, 1])
        L = ellipsoid(np.diag([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]).lift(3, [1, 2, 0])
        np.testing.assert_array_equal(L.q, [3, 1, 2])
        with self.assertRaises(cora.errors.WrongValue):
            self.E.project([2])
        rng = np.random.default_rng(13)
        F = random_ellipsoid(rng, 3)
        M = rng.standard_normal((2, 3))
        self.assertTrue(F.linear_map(M).contains(M @ boundary_points(F, rng, 20), tol=1e-6))

    def test_sums_and_hulls(self):
        # a vector shifts, two ellipsoids add by the enclosure of Halder
        np.testing.assert_array_equal((self.E + np.array([1.0, 1.0])).q, [2, 3])
        np.testing.assert_array_equal((np.array([1.0, 1.0]) + self.E).q, [2, 3])
        A, B = ellipsoid(np.eye(2), [1.0, 0.0]), ellipsoid(4 * np.eye(2), [0.0, 1.0])
        np.testing.assert_allclose(A.plus(B).Q, 9 * np.eye(2), atol=1e-9)
        np.testing.assert_allclose((A + B).q, [1, 1])
        with self.assertRaises(cora.errors.NotSupported):
            A.plus(B, mode="outer")
        # the difference with a vector and with a point
        np.testing.assert_array_equal(self.E.mink_diff(np.array([1.0, 1.0])).q, [0, 1])
        self.assertEqual(self.E.mink_diff(ellipsoid(np.zeros((2, 2)), [1.0, 1.0])).q.tolist(), [0.0, 1.0])
        # unions and equality
        small = ellipsoid(0.01 * np.eye(2), [1.0, 2.0])
        self.assertTrue(self.E.unite(ellipsoid(np.zeros((2, 2)), [1.0, 2.0])).isequal(self.E))
        self.assertTrue(self.E.convex_hull(ellipsoid(np.zeros((2, 2)), [1.0, 2.0])).isequal(self.E))
        with self.assertRaises(cora.errors.NotSupported):
            self.E.unite(ellipsoid(0.01 * np.eye(2), [9.0, 9.0]))
        self.assertTrue(self.E.isequal(ellipsoid([[4.0, 0.0], [0.0, 1.0]], [1.0, 2.0])))
        self.assertFalse(self.E.isequal(small))
        # the sum holds the sums of points of the summands
        rng = np.random.default_rng(17)
        F, G = random_ellipsoid(rng, 3), random_ellipsoid(rng, 3)
        sums = boundary_points(F, rng, 20) + boundary_points(G, rng, 20)
        self.assertTrue(F.plus(G).contains(sums, tol=1e-6))

    def test_distance_and_intersection(self):
        # -1 at the center, 0 on the boundary, positive outside
        d = self.E.distance(np.array([[1.0, 3.0, 1.0], [2.0, 2.0, 4.0]]))
        np.testing.assert_allclose(d, [[-1.0, 0.0, 3.0]], atol=1e-12)
        point = ellipsoid(np.zeros((2, 2)), [1.0, 3.0])
        self.assertTrue(self.E.is_intersecting(point))
        self.assertTrue(self.E.is_intersecting(np.array([1.0, 2.0])))
        self.assertFalse(self.E.is_intersecting(np.array([5.0, 5.0])))
        with self.assertRaises(cora.errors.NotSupported):
            self.E.is_intersecting(self.E)

    def test_conversions_and_samples(self):
        # the parallelotope around the ellipsoid, and its refused norm modes
        Z = self.E.zonotope()
        np.testing.assert_allclose(Z.generators, [[2, 0], [0, 1]], atol=1e-12)
        rng = np.random.default_rng(19)
        F = random_ellipsoid(rng, 3)
        self.assertTrue(F.zonotope().contains(boundary_points(F, rng, 10), tol=1e-8))
        with self.assertRaises(cora.errors.NotSupported):
            F.zonotope("outer:norm")
        # random points, the box and the zonotope converted
        points = F.rand_point(20, cora.Rng(7))
        self.assertTrue(F.contains(points, tol=1e-6))
        self.assertTrue(F.contains(F.rand_point(20, cora.Rng(8), type="extreme"), tol=1e-6))
        # the ellipsoids of a box and of a zonotope
        box = cora.Interval(np.array([0.0, 0.0]), np.array([2.0, 1.0]))
        np.testing.assert_allclose(cora.Ellipsoid(box).Q, 2 * np.diag([1.0, 0.25]), atol=1e-12)
        np.testing.assert_allclose(cora.Ellipsoid(box).q, [1.0, 0.5])
        square = cora.Zonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 2.0]]))
        np.testing.assert_allclose(cora.Ellipsoid(square).Q, 2 * np.diag([1.0, 4.0]))
        np.testing.assert_allclose(cora.Ellipsoid(square, "inner:norm").Q, np.diag([1.0, 4.0]))
        with self.assertRaises(cora.errors.NotSupported):
            cora.Ellipsoid(cora.Zonotope(np.zeros(2), rng.standard_normal((2, 3))))
        # the outline, the surface points and a random ellipsoid
        outline = ellipsoid([[4.0, 0.0], [0.0, 1.0]]).vertices()[0]
        self.assertEqual(outline.shape, (256, 2))
        self.assertEqual(ellipsoid(np.eye(3)).vertices_3d()[0].shape, (200, 3))
        R = cora.Ellipsoid.generate_random(cora.Rng(23), dimension=4, center=np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(R.q, [1, 2, 3, 4])
        flat = cora.Ellipsoid.generate_random(cora.Rng(29), dimension=4, is_degenerate=True)
        self.assertFalse(flat.is_full_dim())


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
