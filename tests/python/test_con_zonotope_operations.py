"""test_con_zonotope_operations - ConZonotope in Python: support function, vertices, sampling and conversions.

CORA authors: Dmitry Grebenyuk, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@conZonotope/dim.m
# CORA: contSet/@conZonotope/generateRandom.m
# CORA: contSet/@conZonotope/generators.m
# CORA: contSet/@conZonotope/polytope.m
# CORA: contSet/@conZonotope/randPoint_.m
# CORA: contSet/@conZonotope/supportFunc_.m
# CORA: contSet/@conZonotope/vertices_.m
# CORA: contSet/@conZonotope/zonoBundle.m
# CORA: contSet/@conZonotope/zonotope.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def hexagon():
    """The set {(a, b) | |a| <= 1, |b| <= 1, |a + b| <= 1}: a cube cut by a plane, in the plane."""
    return cora.ConZonotope(np.zeros(2), np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
                            np.array([[1.0, 1.0, 1.0]]), np.array([0.0]))


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class ConZonotopeOperations(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_dim_and_generators(self):
        # the dimension is the rows of the center, the generators are as given
        H = hexagon()
        self.assertEqual(H.dim, 2)
        np.testing.assert_array_equal(H.generators, [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    def test_support_function(self):
        # the formula without constraints, a linear program with them, -Inf if empty
        plain = cora.ConZonotope(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 2.0]]))
        self.assertAlmostEqual(plain.support_function(np.array([1.0, 1.0])), 6.0)
        H = hexagon()
        self.assertAlmostEqual(H.support_function(np.array([1.0, 1.0])), 1.0)
        self.assertAlmostEqual(H.support_function(np.array([1.0, -1.0])), 2.0)
        self.assertLess(cora.ConZonotope.empty(2).support_function(np.array([1.0, 0.0])), -1e300)
        with self.assertRaises(cora.errors.DimensionMismatch):
            H.support_function(np.array([1.0]))

    def test_vertices(self):
        # the corners of the hexagon in the plane, of a cut cube in space
        corners = hexagon().vertices()[0]
        self.assertEqual(corners.shape, (6, 2))
        for corner in ([1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]):
            self.assertTrue(np.any(np.all(np.isclose(corners, corner), axis=1)))
        plane = cora.ConZonotope(np.zeros(3), np.eye(3), np.array([[1.0, 1.0, 1.0]]), np.array([0.0]))
        self.assertEqual(plane.vertices_3d()[0].shape, (6, 3))
        self.assertEqual(cora.ConZonotope(np.zeros(2), np.eye(2)).vertices()[0].shape, (4, 2))
        self.assertEqual(cora.ConZonotope.empty(2).vertices()[0].shape[0], 0)

    def test_rand_point_and_the_conversions(self):
        # sampled points and the corners are inside every conversion
        rng = cora.Rng(95)
        H = hexagon()
        for kind in ("standard", "extreme"):
            points = H.rand_point(8, rng, kind)
            self.assertEqual(points.shape, (2, 8))
            self.assertTrue(H.contains(points, tol=1e-8))
        self.assertTrue(H.contains(H.rand_point(3, rng)))
        with self.assertRaises(cora.errors.WrongValue):
            H.rand_point(2, rng, "uniform")
        # every conversion contains the corners of the hexagon
        corners = np.array([[1, 1, 0, -1, -1, 0], [0, -1, -1, 0, 1, 1]], dtype=float)
        Z = H.zonotope()
        self.assertIsInstance(Z, cora.Zonotope)
        self.assertTrue(Z.contains(corners))
        self.assertTrue(H.zonotope("reduce").contains(corners))
        bundle = H.zono_bundle()
        self.assertIsInstance(bundle, cora.ZonoBundle)
        self.assertTrue(bundle.contains(corners))
        P = H.polytope()
        self.assertIsInstance(P, cora.Polytope)
        self.assertTrue(P.contains(corners))
        self.assertFalse(P.contains(np.array([1.0, 1.0])))
        with self.assertRaises(cora.errors.EmptySet):
            cora.ConZonotope.empty(2).zonotope()
        with self.assertRaises(cora.errors.WrongValue):
            H.zonotope("pca")

    def test_generate_random(self):
        # the sizes, the constraint count and the dimension of a center
        rng = cora.Rng(96)
        cz = cora.ConZonotope.generate_random(rng, dimension=3, nr_generators=6, nr_constraints=2)
        self.assertEqual((cz.dim, cz.generators.shape, cz.A.shape), (3, (3, 6), (2, 6)))
        self.assertFalse(cz.representsa("emptySet"))
        plain = cora.ConZonotope.generate_random(rng, dimension=2, nr_generators=3, nr_constraints=0)
        self.assertEqual(plain.A.shape[0], 0)
        centered = cora.ConZonotope.generate_random(rng, center=np.array([1.0, 2.0]))
        self.assertEqual(centered.dim, 2)
        with self.assertRaises(cora.errors.WrongValue):
            cora.ConZonotope.generate_random(rng, dimension=0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
