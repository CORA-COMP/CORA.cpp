"""test_poly_zonotope - the PolyZonotope class in Python: constructors, queries and set operations.

CORA authors: Niklas Kochdumper, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@polyZonotope/polyZonotope.m
# CORA: contSet/@zonotope/polyZonotope.m
# CORA: contSet/@interval/polyZonotope.m
# CORA: contSet/@polyZonotope/dim.m
# CORA: contSet/@polyZonotope/center.m
# CORA: contSet/@polyZonotope/zonotope.m
# CORA: contSet/@polyZonotope/interval.m
# CORA: contSet/@polyZonotope/supportFunc_.m
# CORA: contSet/@polyZonotope/project.m
# CORA: contSet/@polyZonotope/display.m
# CORA: contSet/@polyZonotope/plus.m
# CORA: contSet/@polyZonotope/exactPlus.m
# CORA: contSet/@polyZonotope/mtimes.m
# CORA: contSet/@polyZonotope/cartProd_.m
# CORA: contSet/@polyZonotope/enclose.m
# CORA: contSet/@polyZonotope/reduce.m
# CORA: contSet/@polyZonotope/quadMap.m
# CORA: contSet/@polyZonotope/randPoint_.m
# CORA: contSet/@polyZonotope/compact_.m
# CORA: contSet/@polyZonotope/isIntersecting_.m
# CORA: contSet/@polyZonotope/splitDepFactor.m
# CORA: contSet/@polyZonotope/splitLongestGen.m
# CORA: contSet/@polyZonotope/cartProd_.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# ----------------------------------------  BEGIN CODE  ---------------------------------------- #

# ----------------------------------------  AUXILIARY  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def known():
    """(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t on the factors a (id 1) and b (id 2)."""
    return cora.PolyZonotope(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0], [1.0, 2.0]]),
        np.array([[0.0], [1.0]]),
        np.array([[1, 1], [0, 1]]),
        [1, 2],
    )


# ===========================================  MAIN  =========================================== #


# ============================================  MAIN  ============================================ #

class Constructor(unittest.TestCase):
    # the forms of the constructor and what they hold
    def test_forms(self):
        P = known()
        self.assertEqual(P.dim, 2)
        self.assertEqual(P.class_name, "PolyZonotope")
        np.testing.assert_array_equal(P.c, [1, 0])
        self.assertEqual(P.G.shape, (2, 2))
        self.assertEqual(P.E.shape, (2, 2))
        self.assertEqual(list(P.id), [1, 2])
        # a zonotope and a box convert with E = I
        Z = cora.PolyZonotope(cora.Zonotope(np.array([1.0, 2.0]), np.eye(2)))
        np.testing.assert_array_equal(Z.E, np.eye(2, dtype=int))
        box = cora.PolyZonotope(cora.Interval(np.array([0.0, 0.0]), np.array([2.0, 4.0])))
        np.testing.assert_allclose(box.c, [1, 2])

    # equal exponents add
    def test_merge(self):
        P = cora.PolyZonotope(np.zeros(1), np.array([[1.0, 2.0]]), None, np.array([[1, 1]]), [1])
        self.assertEqual(P.G.shape, (1, 1))
        np.testing.assert_allclose(P.G, [[3.0]])

    # generators of another dimension than the center are refused
    def test_refusals(self):
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.PolyZonotope(np.zeros(1), np.zeros((2, 1)), None, np.array([[1]]), [1])


class Operations(unittest.TestCase):
    def test_queries(self):
        P = known()
        box = P.interval()
        np.testing.assert_allclose(box.inf, [0, -4])
        np.testing.assert_allclose(box.sup, [2, 4])
        self.assertEqual(P.zonotope().class_name, "Zonotope")
        self.assertGreaterEqual(P.support_function(np.array([1.0, 0.0])), 2.0 - 1e-12)
        self.assertEqual(P.project([1]).dim, 1)
        self.assertIn("polyZonotope", P.display())

    # the sums and the maps keep or free the factors
    def test_sums_and_maps(self):
        P = known()
        Z = cora.Zonotope(np.array([1.0, 1.0]), np.array([[1.0], [0.0]]))
        np.testing.assert_allclose(P.plus(Z).c, [2, 1])
        self.assertEqual(P.plus(P).dim, 2)
        zero = P.exact_plus(P.linear_map(-np.eye(2)))
        self.assertEqual(zero.G.shape[1], 0)
        mapped = P.linear_map(2 * np.eye(2))
        np.testing.assert_allclose(mapped.c, [2, 0])
        widened = P.linear_map(cora.IntervalMatrix(np.eye(2), 0.1 * np.eye(2)))
        self.assertEqual(widened.GI.shape[1], 3)
        self.assertEqual(P.cart_prod(Z).dim, 4)
        self.assertEqual(P.enclose(P.linear_map(np.zeros((2, 2)))).dim, 2)

    # reduce to a box, and the quadratic map
    def test_reduce_and_quad_map(self):
        P = known()
        R = P.reduce(1)
        self.assertEqual(R.G.shape[1], 0)
        self.assertEqual(R.GI.shape[1], 2)
        Q = P.quad_map([np.array([[0.0, 1.0], [0.0, 0.0]])])
        self.assertEqual(Q.dim, 1)
        self.assertEqual(P.compact().dim, 2)

    # c = 1, 2 a + 3 a^2 halved along a: 2.75 + 2.5 b + 0.75 b^2 and 0.75 - 0.5 b + 0.75 b^2
    def test_split(self):
        P = cora.PolyZonotope(np.array([1.0]), np.array([[2.0, 3.0]]), np.zeros((1, 0)),
                              np.array([[1, 2]]), [1])
        P1, P2 = P.split_dep_factor(1)
        np.testing.assert_allclose(np.asarray(P1.c).ravel(), [2.75])
        np.testing.assert_allclose(np.asarray(P1.G), [[2.5, 0.75]])
        np.testing.assert_allclose(np.asarray(P2.c).ravel(), [0.75])
        np.testing.assert_allclose(np.asarray(P2.G), [[-0.5, 0.75]])
        L1, L2 = P.split_longest_gen()
        np.testing.assert_allclose(np.asarray(L1.G), np.asarray(P1.G))
        with self.assertRaises(cora.errors.WrongValue):
            P.split_dep_factor(7)

    # sampled points lie in the zonotope, and a polytope far away is missed
    def test_points_and_polytope(self):
        P = known()
        rng = cora.Rng(1)
        points = P.rand_point(30, rng)
        self.assertEqual(points.shape, (2, 30))
        self.assertTrue(P.zonotope().contains(points))
        miss = cora.Polytope(np.array([[1.0, 0.0]]), np.array([-1.0]))
        hit = cora.Polytope(np.array([[1.0, 0.0]]), np.array([1.0]))
        self.assertFalse(P.is_intersecting(miss))
        self.assertTrue(P.is_intersecting(hit))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------  END OF CODE  ---------------------------------------- #

# ----------------------------------------  END OF CODE  ----------------------------------------- #
