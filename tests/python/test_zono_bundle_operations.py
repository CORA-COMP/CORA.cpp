"""test_zono_bundle_operations - ZonoBundle in Python: maps, sums, conversions, queries and sampling.

CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@interval/zonoBundle.m
# CORA: contSet/@zonoBundle/dim.m
# CORA: contSet/@zonoBundle/generateRandom.m
# CORA: contSet/@zonoBundle/mtimes.m
# CORA: contSet/@zonoBundle/or.m
# CORA: contSet/@zonoBundle/origin.m
# CORA: contSet/@zonoBundle/plus.m
# CORA: contSet/@zonoBundle/polytope.m
# CORA: contSet/@zonoBundle/project.m
# CORA: contSet/@zonoBundle/quadMap.m
# CORA: contSet/@zonoBundle/randPoint_.m
# CORA: contSet/@zonoBundle/reduce.m
# CORA: contSet/@zonoBundle/replace.m
# CORA: contSet/@zonoBundle/representsa_.m
# CORA: contSet/@zonoBundle/split.m
# CORA: contSet/@zonoBundle/supportFunc_.m
# CORA: contSet/@zonoBundle/vertices_.m
# CORA: contSet/@zonoBundle/zonotope.m
# CORA: contSet/@zonotope/zonoBundle.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def two_zonotopes():
    """The box [-2, 2] x [-1, 1] and a hexagon in [-1, 3] x [-1, 1]: their boxes meet in [-1, 2] x [-1, 1]."""
    return cora.ZonoBundle([zonotope([0.0, 0.0], [[2.0, 0.0], [0.0, 1.0]]),
                            zonotope([1.0, 0.0], [[1.0, 1.0], [0.0, 1.0]])])


def common_point_bundle(rng, n, k, m):
    """A bundle of k zonotopes that all contain one point (given back)."""
    x0 = rng.standard_normal(n)
    members = []
    for _ in range(k):
        G = rng.standard_normal((n, m))
        members.append(zonotope(x0 - G @ rng.uniform(-1, 1, m), G))
    return cora.ZonoBundle(members), x0


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class Conversions(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_origin_and_the_conversions_of_one_set(self):
        # a bundle of one member from a point, a zonotope or a box
        O = cora.ZonoBundle.origin(3)
        self.assertEqual((O.dim, O.parallel_sets), (3, 1))
        self.assertEqual(two_zonotopes().dim, 2)
        self.assertTrue(O.representsa("origin"))
        with self.assertRaises(cora.errors.WrongValue):
            cora.ZonoBundle.origin(0)
        Z = zonotope([1.0, 2.0], [[1.0, 0.0], [0.5, 1.0]])
        fromZ = cora.ZonoBundle(Z)
        self.assertEqual(fromZ.parallel_sets, 1)
        np.testing.assert_array_equal(fromZ.zonotopes[0].generators, Z.generators)
        fromI = cora.ZonoBundle(cora.Interval(np.array([0.0, -1.0]), np.array([2.0, 1.0])))
        np.testing.assert_array_equal(fromI.zonotopes[0].center, [1.0, 0.0])
        with self.assertRaises(cora.errors.NotSupported):
            cora.ZonoBundle(cora.Interval.empty(2))

    def test_zonotope_polytope_and_representsa(self):
        # the member of least volume, the halfspace form and the type tests
        zB = two_zonotopes()
        small = zonotope([0.5, 0.0], [[0.5, 0.0], [0.0, 0.25]])
        self.assertAlmostEqual(cora.ZonoBundle([zB.zonotopes[0], small]).zonotope().center[0], 0.5)
        P = zB.polytope()
        self.assertTrue(P.contains(np.array([2.0, 0.0])))
        self.assertFalse(P.contains(np.array([2.5, 0.0])))
        self.assertTrue(zB.representsa("polytope") and not zB.representsa("origin"))
        self.assertTrue(cora.ZonoBundle.empty(2).representsa("emptySet"))
        self.assertTrue(cora.ZonoBundle([zonotope([0.0, 0.0], np.eye(2))] * 2).representsa("interval"))
        with self.assertRaises(cora.errors.NotSupported):
            zB.representsa("point")


class MapsAndSums(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_linear_map_and_the_operators(self):
        # the image under a matrix, by method and by the operators
        zB = two_zonotopes()
        M = np.array([[1.0, 0.0], [0.0, 2.0]])
        R = zB.linear_map(M)
        np.testing.assert_array_equal(R.zonotopes[0].generators, [[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal((M @ zB).zonotopes[1].center, [1.0, 0.0])
        np.testing.assert_array_equal((2.0 * zB).zonotopes[1].generators, [[2.0, 2.0], [0.0, 2.0]])
        with self.assertRaises(cora.errors.DimensionMismatch):
            zB.linear_map(np.ones((1, 3)))

    def test_plus_with_points_and_sets(self):
        # a point moves the centers, a bundle gives the pairwise sums
        zB = two_zonotopes()
        np.testing.assert_array_equal(zB.plus(np.array([1.0, 2.0])).zonotopes[1].center, [2.0, 2.0])
        np.testing.assert_array_equal((zB + np.array([1.0, 2.0])).zonotopes[0].center, [1.0, 2.0])
        np.testing.assert_array_equal((np.array([1.0, 2.0]) + zB).zonotopes[0].center, [1.0, 2.0])
        W = zonotope([0.0, 0.0], [[1.0], [0.0]])
        self.assertEqual(zB.plus(W).zonotopes[0].generators.shape, (2, 3))
        self.assertEqual((zB + cora.Interval(np.zeros(2), np.ones(2))).parallel_sets, 2)
        W2 = zonotope([5.0, 5.0], [[1.0], [1.0]])
        pairs = zB.plus(cora.ZonoBundle([W, W2]))
        self.assertEqual(pairs.parallel_sets, 4)
        np.testing.assert_array_equal(pairs.zonotopes[3].center, [6.0, 5.0])
        self.assertIsInstance(cora.plus(zB, W), cora.ZonoBundle)
        self.assertTrue(zB.plus(cora.Interval.empty(2)).representsa("emptySet"))
        with self.assertRaises(cora.errors.DimensionMismatch):
            zB.plus(np.array([1.0]))

    def test_project_quad_map_reduce_and_replace(self):
        # every member is projected, mapped, reduced or replaced
        zB = two_zonotopes()
        one = zB.project([1])
        self.assertEqual((one.dim, one.parallel_sets), (1, 2))
        Q = [np.eye(2)]
        mapped = zB.quad_map(Q)
        self.assertEqual((mapped.dim, mapped.parallel_sets), (1, 2))
        with self.assertRaises(cora.errors.DimensionMismatch):
            zB.quad_map([np.zeros((3, 3))])
        # a reduced bundle has few generators per member and keeps its common point
        rng = np.random.default_rng(91)
        big, x0 = common_point_bundle(rng, 2, 3, 7)
        reduced = big.reduce(2)
        self.assertTrue(all(Z.generators.shape[1] <= 4 for Z in reduced.zonotopes))
        self.assertTrue(reduced.contains(x0))
        with self.assertRaises(cora.errors.NotSupported):
            big.reduce(1, "pca")
        W = zonotope([3.0, 3.0], [[1.0], [1.0]])
        replaced = zB.replace(1, W)
        np.testing.assert_array_equal(replaced.zonotopes[1].center, [3.0, 3.0])
        np.testing.assert_array_equal(replaced.zonotopes[0].center, [0.0, 0.0])
        with self.assertRaises(cora.errors.WrongValue):
            zB.replace(2, W)

    def test_properties_of_the_maps_on_random_bundles(self):
        # the image of a common point is in the image
        rng = np.random.default_rng(92)
        for _ in range(3):
            zB, x0 = common_point_bundle(rng, 3, 2, 4)
            M = rng.standard_normal((2, 3))
            self.assertTrue(zB.linear_map(M).contains(M @ x0))
            w = rng.standard_normal(3)
            self.assertTrue(zB.plus(w).contains(x0 + w))
            self.assertTrue(zB.project([2, 0]).contains(x0[[2, 0]]))
            Q = rng.standard_normal((3, 3))
            self.assertTrue(zB.quad_map([Q]).contains(np.array([x0 @ Q @ x0])))


class QueriesAndSampling(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_support_function_and_vertices(self):
        # the box [-1, 2] x [-1, 1] and two unit cubes that meet in a cube
        zB = cora.ZonoBundle([zonotope([0.0, 0.0], [[2.0, 0.0], [0.0, 1.0]]),
                              zonotope([1.0, 0.0], [[2.0, 0.0], [0.0, 1.0]])])
        self.assertAlmostEqual(zB.support_function(np.array([1.0, 1.0])), 3.0)
        self.assertAlmostEqual(zB.support_function(np.array([-1.0, 0.0])), 1.0)
        self.assertLess(cora.ZonoBundle.empty(2).support_function(np.array([1.0, 0.0])), -1e300)
        corners = zB.vertices()[0]
        self.assertEqual(corners.shape, (4, 2))
        self.assertAlmostEqual(corners[:, 0].max(), 2.0)
        self.assertAlmostEqual(corners[:, 0].min(), -1.0)
        cube = np.eye(3)
        cubes = cora.ZonoBundle([zonotope(np.zeros(3), cube), zonotope(np.ones(3), cube)])
        self.assertEqual(cubes.vertices_3d()[0].shape, (8, 3))
        with self.assertRaises(cora.errors.DimensionMismatch):
            zB.support_function(np.array([1.0]))

    def test_rand_point_split_unite_and_generate_random(self):
        # sampling, the pieces of a split, a union and a random bundle
        rng = cora.Rng(93)
        zB = two_zonotopes()
        for kind in ("standard", "extreme"):
            points = zB.rand_point(6, rng, kind)
            self.assertEqual(points.shape, (2, 6))
            self.assertTrue(zB.contains(points))
        self.assertTrue(zB.contains(zB.rand_point(3, rng)))
        # the pieces are cut at the middle of the box [-1, 2] x [-1, 1]
        low, high = zB.split(0)
        self.assertEqual((low.parallel_sets, high.parallel_sets), (3, 3))
        self.assertAlmostEqual(low.interval().sup[0], 0.5)
        self.assertAlmostEqual(high.interval().inf[0], 0.5)
        self.assertEqual(len(zB.split()), 2)
        # the union of two bundles is one zonotope without constraints
        united = zB.unite(zonotope([5.0, 5.0], np.eye(2)))
        self.assertEqual(united.parallel_sets, 1)
        self.assertTrue(united.contains(np.array([5.0, 5.0])) and united.contains(np.array([0.0, 0.0])))
        self.assertTrue(zB.unite(zB).contains(np.array([2.0, 0.0])))
        self.assertTrue(zB.unite(cora.Interval(np.array([4.0, 4.0]), np.array([5.0, 5.0]))).dim == 2)
        random = cora.ZonoBundle.generate_random(rng, 3, 4)
        self.assertEqual((random.dim, random.parallel_sets), (3, 4))
        self.assertFalse(random.representsa("emptySet"))
        with self.assertRaises(cora.errors.WrongValue):
            zB.rand_point(-1, rng)
        with self.assertRaises(cora.errors.DimensionMismatch):
            zB.unite(zonotope([0.0], [[1.0]]))
        with self.assertRaises(cora.errors.EmptySet):
            cora.ZonoBundle.empty(2).split(0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
