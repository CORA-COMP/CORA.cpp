"""test_zono_bundle - the ZonoBundle class in Python: constructor, queries and set operations.

CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       04-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@zonoBundle/and_.m
# CORA: contSet/@zonoBundle/cartProd_.m
# CORA: contSet/@zonoBundle/center.m
# CORA: contSet/@zonoBundle/conZonotope.m
# CORA: contSet/@zonoBundle/contains_.m
# CORA: contSet/@zonoBundle/display.m
# CORA: contSet/@zonoBundle/empty.m
# CORA: contSet/@zonoBundle/enclose.m
# CORA: contSet/@zonoBundle/enlarge.m
# CORA: contSet/@zonoBundle/interval.m
# CORA: contSet/@zonoBundle/isBounded.m
# CORA: contSet/@zonoBundle/isFullDim.m
# CORA: contSet/@zonoBundle/isIntersecting_.m
# CORA: contSet/@zonoBundle/isequal.m
# CORA: contSet/@zonoBundle/lift_.m
# CORA: contSet/@zonoBundle/zonoBundle.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def zonotope(c, G):
    return cora.Zonotope(np.array(c, dtype=float), np.array(G, dtype=float))


def squares():
    """The squares [-1, 1]^2 and [0, 2]^2, which meet in [0, 1]^2."""
    return cora.ZonoBundle([zonotope([0.0, 0.0], np.eye(2)), zonotope([1.0, 1.0], np.eye(2))])


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class ZonoBundle(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_constructor_and_the_queries(self):
        zB = squares()
        self.assertEqual(zB.dim, 2)
        self.assertEqual(zB.parallel_sets, 2)
        self.assertEqual(zB.class_name, "ZonoBundle")
        np.testing.assert_array_equal(zB.zonotopes[1].center, [1.0, 1.0])
        self.assertTrue(zB.is_bounded())
        self.assertTrue(zB.is_full_dim())
        self.assertIn("zonotope bundle", repr(zB))
        self.assertIn("zonotopes: 2", str(zB))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.ZonoBundle([])
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.ZonoBundle([zonotope([0.0], [[1.0]]), zonotope([0.0, 0.0], np.eye(2))])
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.ZonoBundle([1.0])

    def test_empty_conversion_and_center(self):
        # the empty bundle
        E = cora.ZonoBundle.empty(3)
        self.assertEqual(E.dim, 3)
        self.assertEqual(E.parallel_sets, 2)
        self.assertEqual(E.interval().inf.shape[-1], 0)
        self.assertEqual(E.center.shape[-1], 0)
        self.assertFalse(E.is_full_dim())
        with self.assertRaises(cora.errors.WrongValue):
            cora.ZonoBundle.empty(0)
        cz = squares().con_zonotope()
        self.assertIsInstance(cz, cora.ConZonotope)
        box = cz.interval()
        np.testing.assert_allclose(box.inf, [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(box.sup, [1.0, 1.0], atol=1e-9)
        self.assertTrue(squares().contains(squares().center))

    def test_interval_and_the_set_interface(self):
        box = squares().interval()
        np.testing.assert_array_equal(box.inf, [0.0, 0.0])
        np.testing.assert_array_equal(box.sup, [1.0, 1.0])
        lifted = squares().lift(2, [1, 0])
        self.assertIsInstance(lifted, cora.ZonoBundle)
        # enlarging scales the members
        enlarged = squares().enlarge(2.0)
        np.testing.assert_array_equal(enlarged.zonotopes[0].generators, 2.0 * np.eye(2))
        self.assertEqual(squares().vertices()[0].shape, (4, 2))

    def test_intersect_and_cart_prod(self):
        zB = squares()
        Z = zonotope([0.0, 1.0], [[1.0], [1.0]])
        self.assertEqual(zB.intersect(Z).parallel_sets, 3)
        self.assertEqual(zB.intersect(cora.Interval(np.zeros(2), np.ones(2))).parallel_sets, 3)
        self.assertEqual(zB.intersect(zB).parallel_sets, 4)
        self.assertIsInstance(cora.intersect(zB, Z), cora.ZonoBundle)
        self.assertEqual((zB & Z).parallel_sets, 3)
        P = zB.cart_prod(zB)
        self.assertEqual((P.dim, P.parallel_sets), (4, 4))
        self.assertEqual(zB.cart_prod(np.array([5.0])).dim, 3)
        self.assertEqual(zB.cart_prod(Z).dim, 4)
        self.assertEqual(cora.cart_prod(zB, cora.Interval(np.zeros(1), np.ones(1))).dim, 3)
        with self.assertRaises(cora.errors.DimensionMismatch):
            zB.intersect(zonotope([0.0], [[1.0]]))

    def test_enclose(self):
        # enclose with a zonotope
        zB = squares()
        R = zB.enclose(zonotope([4.0, 0.0], [[1.0], [0.5]]))
        self.assertEqual(R.parallel_sets, 2)
        S = zB.enclose(cora.ZonoBundle([zonotope([4.0, 0.0], np.eye(2))] * 2))
        self.assertEqual(S.parallel_sets, 2)
        M = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.assertEqual(zB.enclose(M, zonotope([0.5, 0.5], [[0.1], [0.0]])).parallel_sets, 2)
        shift = cora.ZonoBundle([zonotope([0.5, 0.5], np.eye(2))] * 3)
        self.assertEqual(zB.enclose(M, shift).parallel_sets, 2)
        self.assertIsInstance(zB.enclose(zonotope([0.0, 0.0], np.eye(2))), cora.ZonoBundle)
        with self.assertRaises(cora.errors.WrongValue):
            zB.enclose(cora.ZonoBundle([zonotope([0.0, 0.0], np.eye(2))]))

    def test_contains_intersection_tests_and_equality(self):
        zB = squares()
        self.assertTrue(zB.contains(np.array([0.5, 0.5])))
        self.assertFalse(zB.contains(np.array([-0.5, -0.5])))
        self.assertTrue(np.array([0.5, 0.5]) in zB)
        self.assertTrue(zB.contains(np.array([[0.5, 0.25], [0.5, 0.75]])))
        self.assertTrue(zB.contains(zonotope([0.5, 0.5], 0.25 * np.eye(2))))
        self.assertTrue(zB.contains(cora.Interval(np.array([0.25, 0.25]), np.array([0.75, 0.75]))))
        # containment and intersection of a bundle
        self.assertTrue(zB.contains(zB.intersect(zonotope([0.5, 0.5], 0.25 * np.eye(2)))))
        self.assertTrue(cora.contains(zB, zonotope([0.5, 0.5], 0.25 * np.eye(2))))
        self.assertTrue(zB.is_intersecting(zonotope([0.5, 0.5], 0.25 * np.eye(2))))
        self.assertFalse(zB.is_intersecting(zonotope([-0.75, -0.75], 0.25 * np.eye(2))))
        self.assertTrue(zB.is_intersecting(zB, type="approx"))
        self.assertTrue(zB.is_intersecting(np.array([0.5, 0.5])))
        self.assertTrue(cora.is_intersecting(zB, cora.Interval(np.array([0.9, 0.9]), np.array([3.0, 3.0]))))
        # equal bundles: the members in any order
        self.assertTrue(zB.isequal(cora.ZonoBundle(list(reversed(zB.zonotopes)))))
        self.assertFalse(zB.isequal(cora.ZonoBundle([zB.zonotopes[0]])))
        self.assertTrue(cora.isequal(zB, zB))
        self.assertTrue(zB == cora.ZonoBundle(list(reversed(zB.zonotopes))))
        with self.assertRaises(cora.errors.NotSupported):
            zB.contains(np.array([0.5, 0.5]), method="exact:polytope")
        with self.assertRaises(cora.errors.NoSpecificAlg):
            zB.contains(np.array([0.5, 0.5]), method="nothing")
        with self.assertRaises(cora.errors.WrongValue):
            zB.is_intersecting(zB, type="fast")
        with self.assertRaises(cora.errors.WrongValue):
            zB.isequal(zB, tol=-1.0)


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
