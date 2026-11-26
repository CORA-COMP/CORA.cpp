"""test_con_poly_zono - the ConPolyZono and PolyLevelSet classes in Python: the bindings.

CORA authors: Niklas Kochdumper
Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@conPolyZono/conPolyZono.m
# CORA: contSet/@conPolyZono/origin.m
# CORA: contSet/@conPolyZono/empty.m
# CORA: contSet/@conPolyZono/dim.m
# CORA: contSet/@conPolyZono/center.m
# CORA: contSet/@conPolyZono/mtimes.m
# CORA: contSet/@conPolyZono/plus.m
# CORA: contSet/@conPolyZono/exactPlus.m
# CORA: contSet/@conPolyZono/cartProd_.m
# CORA: contSet/@conPolyZono/and_.m
# CORA: contSet/@conPolyZono/linComb.m
# CORA: contSet/@conPolyZono/convHull_.m
# CORA: contSet/@conPolyZono/project.m
# CORA: contSet/@conPolyZono/lift_.m
# CORA: contSet/@conPolyZono/compact_.m
# CORA: contSet/@conPolyZono/reduce.m
# CORA: contSet/@conPolyZono/split.m
# CORA: contSet/@conPolyZono/getSubset.m
# CORA: contSet/@conPolyZono/splitDepFactor.m
# CORA: contSet/@conPolyZono/splitLongestGen.m
# CORA: contSet/@conPolyZono/interval.m
# CORA: contSet/@conPolyZono/zonotope.m
# CORA: contSet/@conPolyZono/polyZonotope.m
# CORA: contSet/@conPolyZono/supportFunc_.m
# CORA: contSet/@conPolyZono/isequal.m
# CORA: contSet/@conPolyZono/representsa_.m
# CORA: contSet/@conPolyZono/display.m
# CORA: contSet/@conPolyZono/vertices_.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Bindings(unittest.TestCase):
    # the constructor, the fields and the conversions
    def test_fields(self):
        S = known()
        self.assertEqual(S.dim, 2)
        self.assertEqual(S.class_name, "ConPolyZono")
        np.testing.assert_array_equal(S.c, [1, 0])
        self.assertEqual(S.E.shape, (2, 2))
        self.assertEqual(S.EC.shape, (2, 2))
        self.assertEqual(list(S.id), [1, 2])
        self.assertTrue(S.constraints.contains(np.array([[0.5, -0.4]]), tol=1e-12))
        self.assertIsInstance(S.poly_zonotope(), cora.PolyZonotope)
        self.assertIsInstance(S.zonotope(), cora.Zonotope)
        Z = cora.Zonotope(np.array([1.0, 2.0]), np.eye(2))
        self.assertEqual(cora.ConPolyZono(Z).G.shape, (2, 2))
        self.assertEqual(cora.ConPolyZono.origin(2).dim, 2)
        self.assertTrue(cora.ConPolyZono.empty(2).representsa("emptySet"))
        self.assertTrue(cora.ConPolyZono.origin(2).representsa("point"))
        self.assertIn("constrained", S.display())

    # the operations: each returns a set of the class
    def test_operations(self):
        S = known()
        M = np.array([[2.0, 1.0], [-1.0, 0.5]])
        for R in (S.linear_map(M), S.plus(S), S.plus(np.array([1.0, 2.0])), S.exact_plus(S),
                  S.intersect(S), S.linear_combination(S), S.convex_hull(S), S.compact(),
                  S.reduce(1), S.lift(2, [1, 0]), S.get_subset(1, box(0.25, 0.75))):
            self.assertIsInstance(R, cora.ConPolyZono)
        self.assertEqual(S.cart_prod(S).dim, 4)
        self.assertEqual(S.project([1]).dim, 1)
        for halves in (S.split(0), S.split_dep_factor(1), S.split_longest_gen()):
            self.assertEqual(len(halves), 2)
        self.assertTrue(S.isequal(S))
        self.assertFalse(S.representsa("point"))
        I = S.interval()
        self.assertTrue(np.all(I.inf <= S.c) and np.all(S.c <= I.sup))
        # the zonotope's bound in the direction e1 reaches the box's upper end
        self.assertGreaterEqual(float(np.asarray(S.support_function(np.array([1.0, 0.0])))),
                                float(I.sup[0]) - 1e-9)
        self.assertGreater(len(S.vertices()), 0)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def known():
    """(1, 0) + a1 (1, 0.5) + a1 a2^2 (0, 1) + b (0.25, 0) with a1 - 0.5 a2 = 0.7."""
    return cora.ConPolyZono(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.5], [0.0, 1.0]]),
        np.array([[1, 1], [0, 2]]),
        np.array([[1.0, -0.5]]),
        np.array([0.7]),
        np.array([[1, 0], [0, 1]]),
        np.array([[0.25, 0.0]]),
        [1, 2],
    )


def box(lo, hi):
    """The interval [lo, hi] of one dimension."""
    return cora.Interval(np.array([lo]), np.array([hi]))


# ===========================================  MAIN  =========================================== #


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------  END OF CODE  ----------------------------------------- #
