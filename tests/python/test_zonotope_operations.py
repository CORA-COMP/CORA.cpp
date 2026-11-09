"""test_zonotope_operations - abs, box, compact, cart_prod, enclose, convex_hull, intersect, contains

and more of Zonotope in Python, on numpy and Eigen alone.

CORA authors: Matthias Althoff, Niklas Kochdumper, Amr Alanwar, Mark Wetzlinger, Adrian Kulmburg,
              Ivan Brkan
Authors:       Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@zonotope/abs.m
# CORA: contSet/@zonotope/and_.m
# CORA: contSet/@zonotope/boundaryPoint.m
# CORA: contSet/@zonotope/box.m
# CORA: contSet/@zonotope/cartProd_.m
# CORA: contSet/@zonotope/center.m
# CORA: contSet/@zonotope/compact_.m
# CORA: contSet/@zonotope/constrSat.m
# CORA: contSet/@zonotope/contains_.m
# CORA: contSet/@zonotope/convHull_.m
# CORA: contSet/@zonotope/cubMap.m
# CORA: contSet/@zonotope/dH2box.m
# CORA: contSet/@zonotope/dominantDirections.m
# CORA: contSet/@zonotope/enclose.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def diamond():
    """The set |x| + |y| <= 2."""
    return zonotope([0, 0], [[1, 1], [1, -1]])


def random_zonotope(rng, n=3, m=4):
    return zonotope(10 * rng.standard_normal(n), rng.uniform(-1, 1, (n, m)))


def support(Z, rng, count=40):
    """The support function along `count` random directions: (directions, values)."""
    d = rng.standard_normal((count, Z.dim))
    return d, np.array([Z.support_function(x) for x in d])


# ============================================  MAIN  ============================================ #


class Entrywise(unittest.TestCase):
    def test_abs_and_box_and_center(self):
        Z = zonotope([1, -2], [[1, -3, 0.5], [-2, 0.5, 0]])
        np.testing.assert_array_equal(Z.abs().center, [1, 2])
        np.testing.assert_array_equal(Z.abs().generators, [[1, 3, 0.5], [2, 0.5, 0]])
        np.testing.assert_array_equal(Z.box().center, Z.center)
        np.testing.assert_array_equal(Z.box().generators, [[4.5, 0], [0, 2.5]])
        np.testing.assert_array_equal(Z.center, [1, -2])

# zeros and tolerance


class Compact(unittest.TestCase):
    def test_zeros_and_tolerance(self):
        Z = zonotope([1, 2], [[1, 0, 1e-20, 0.5], [0, 0, 0, 0.5]])
        np.testing.assert_array_equal(Z.compact().generators, [[1, 0.5], [0, 0.5]])
        self.assertEqual(Z.compact("zeros", 0.0).generators.shape[1], 3)
        self.assertEqual(Z.compact("zeros", 0.8).generators.shape[1], 1)

    # "all" merges the generators of one direction
    def test_all_merges_aligned_generators(self):
        Z = zonotope([0, 0], [[-0.5, 1, 2, 0], [0, 0, 0, 3]])
        np.testing.assert_allclose(Z.compact("all").generators, [[-3.5, 0], [0, 3]])
        np.testing.assert_allclose(zonotope([0], [[1, -2, 0]]).compact("all").generators, [[3]])
        np.testing.assert_array_equal(Z.compact("aligned").generators, Z.generators)

    def test_errors(self):
        Z = zonotope([0, 0], [[1, 0], [0, 0]])
        with self.assertRaises(ValueError):
            Z.compact("sorted")
        with self.assertRaises(ValueError):
            Z.compact("zeros", -1.0)


class Combining(unittest.TestCase):
    def test_cart_prod_of_each_kind(self):
        Z = zonotope([1, 2], [[1, 2], [3, 4]])
        P = Z.cart_prod(zonotope([5], [[6, 7, 8]]))
        np.testing.assert_array_equal(P.center, [1, 2, 5])
        np.testing.assert_array_equal(P.generators, [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 6, 7, 8]])
        # Cartesian products
        self.assertEqual(Z.cart_prod(np.array([9.0])).generators.shape, (3, 2))
        self.assertEqual(Z.cart_prod(cora.Interval(np.array([0.0, -1.0]), np.array([2.0, 1.0]))).dim, 4)
        self.assertEqual(cora.cart_prod(Z, Z).dim, 4)

    def test_enclose_and_convex_hull(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        Z2 = zonotope([2, 0], [[1], [0]])
        E = Z.enclose(Z2)
        np.testing.assert_array_equal(E.center, [1, 0])
        np.testing.assert_array_equal(E.generators, [[1, -1, 0, 0], [0, 0, 0, 1]])
        rng = np.random.default_rng(4)
        A, B = random_zonotope(rng, 3, 5), random_zonotope(rng, 3, 3)
        box = cora.Interval(np.array([-1.0, 0.0, 0.5]), np.array([2.0, 1.0, 3.0]))
        d = rng.standard_normal((40, 3))
        # the hull bounds the operands
        for hull, other in ((A.convex_hull(B), B), (A.convex_hull(box), box)):
            bound = np.maximum([A.support_function(x) for x in d], [other.support_function(x) for x in d])
            self.assertTrue(np.all([hull.support_function(x) for x in d] >= bound - 1e-9))
        with_point = A.convex_hull(B.center)
        bound = np.maximum([A.support_function(x) for x in d], d @ B.center)
        self.assertTrue(np.all([with_point.support_function(x) for x in d] >= bound - 1e-9))
        self.assertIsInstance(cora.convex_hull(A, B), cora.Zonotope)

    def test_intersect_boxes_and_averaging(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        R = Z.intersect(zonotope([1, 0], [[1, 0], [0, 0.5]]))
        np.testing.assert_array_equal(R.center, [0.5, 0])
        np.testing.assert_array_equal(R.generators, [[0.5, 0], [0, 0.5]])
        self.assertEqual(Z.intersect(cora.Interval(np.array([0.0, -2.0]), np.array([0.5, 2.0]))).dim, 2)
        A = zonotope([1, 0], [[1, 1], [1, -1]])
        B = zonotope([0, 1], [[2, 0], [0, 2]])
        M = A.intersect(B, "averaging")
        np.testing.assert_allclose(M.center, [2 / 3, 1 / 3])
        np.testing.assert_allclose(M.generators[0], [2 / 3, 2 / 3, 2 / 3, 0])
        with self.assertRaises(cora.errors.NotSupported):
            A.intersect(A)
        # an empty intersection
        with self.assertRaises(cora.errors.EmptySet):
            Z.intersect(zonotope([5, 5], [[1, 0], [0, 1]]))


class Constraints(unittest.TestCase):
    def test_constraints_satisfied(self):
        Z = zonotope([0, 0], [[1, 0], [0, 1]])
        C = np.array([[1.0, 1.0]])
        self.assertTrue(Z.constraints_satisfied(C, np.array([3.0])))
        # satisfied constraints
        self.assertFalse(Z.constraints_satisfied(C, np.array([2.0])))
        self.assertFalse(Z.constraints_satisfied(C, np.array([1.9])))


class CubicMap(unittest.TestCase):
    def test_by_hand(self):
        Z = zonotope([1], [[0.5]])
        R = Z.cubic_map([[np.array([[2.0]])]])
        np.testing.assert_allclose(R.center, [2.75])
        np.testing.assert_allclose(R.generators, [[3.0, 0.75, 0.25]])
        M = Z.cubic_map(zonotope([2], [[1]]), zonotope([3], [[0.5]]), [[np.array([[1.0]])]])
        np.testing.assert_allclose(M.center, [6])
        # the cubic map
        np.testing.assert_allclose(M.generators, [[3, 3, 1.5, 1, 0.5, 0.5, 0.25]])

    def test_errors(self):
        Z = zonotope([1, 0], [[0.5], [0.2]])
        with self.assertRaises(ValueError):
            Z.cubic_map([])
        with self.assertRaises(ValueError):
            Z.cubic_map([[np.eye(1)]])


class Measures(unittest.TestCase):
    def test_hausdorff_distance_to_box_methods(self):
        Z = zonotope([0, 0], [[1], [1]])
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("naive"), np.sqrt(2))
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("wgreedy"), 2.0)
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("wopt"), np.sqrt(2))
        self.assertAlmostEqual(Z.hausdorff_distance_to_box("exact"), np.sqrt(2), places=3)
        self.assertEqual(Z.hausdorff_distance_to_box(), Z.hausdorff_distance_to_box("exact"))
        with self.assertRaises(ValueError):
            Z.hausdorff_distance_to_box("fast")
        with self.assertRaises(cora.errors.NotSupported):
            # invalid norms
            Z.hausdorff_distance_to_box("ell")

    def test_dominant_directions(self):
        Z = zonotope([0, 0], [[1, 0, 0.1], [0, 2, 0.1]])
        np.testing.assert_array_equal(Z.dominant_directions(), [[0, 1], [2, 0]])
        np.testing.assert_array_equal(Z.dominant_directions(2, 1), [[0, 1], [2, 0]])
        rng = np.random.default_rng(4)
        Z = random_zonotope(rng, 3, 9)
        P = Z.dominant_directions()
        radius = np.abs(np.linalg.inv(P) @ Z.generators).sum(axis=1)
        for x in rng.standard_normal((30, 3)):
            self.assertGreaterEqual(x @ Z.center + np.abs(P.T @ x) @ radius, Z.support_function(x) - 1e-9)
        with self.assertRaises(ValueError):
            Z.dominant_directions(0, 1)


# points in a diamond
class Containment(unittest.TestCase):
    def test_points_in_a_diamond(self):
        Z = diamond()
        self.assertTrue(Z.contains(np.array([0.0, 0.0])))
        self.assertTrue(np.array([1.0, 1.0]) in Z)
        self.assertTrue(Z.contains(np.array([[0.5, 2.0, -1.0], [0.5, 0.0, 0.5]])))
        self.assertFalse(Z.contains(np.array([[0.5, 1.5], [0.5, 1.5]])))
        self.assertFalse(Z.contains(np.array([1.0 + 1e-10, 1.0])))
        self.assertTrue(Z.contains(np.array([1.0 + 1e-10, 1.0]), tol=1e-6))

    def test_sets_in_a_diamond(self):
        Z = diamond()
        small, large = 0.5 * Z, 1.2 * Z
        for method in ("exact", "exact:venum", "exact:polymax", "approx", "approx:st"):
            self.assertTrue(Z.contains(small, method))
            self.assertFalse(Z.contains(large, method))
        self.assertTrue(Z.contains(cora.Interval(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))))
        self.assertFalse(Z.contains(cora.Interval(np.array([-1.5, -1.5]), np.array([1.5, 1.5]))))
        self.assertTrue(cora.contains(Z, small))
        with self.assertRaises(cora.errors.NotSupported):
            Z.contains(Z, "approx:stDual")
        # an unknown algorithm
        with self.assertRaises(cora.errors.NoSpecificAlg):
            Z.contains(np.array([0.0, 0.0]), "simplex")

    def test_boundary_point(self):
        Z = diamond()
        np.testing.assert_allclose(Z.boundary_point(np.array([1.0, 0.0])), [2, 0], atol=1e-9)
        x = Z.boundary_point(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
        np.testing.assert_allclose(x, [1.5, 0.5], atol=1e-9)
        with self.assertRaises(ValueError):
            Z.boundary_point(np.array([0.0, 0.0]))
        with self.assertRaises(ValueError):
            Z.boundary_point(np.array([1.0, 0.0]), np.array([5.0, 0.0]))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
