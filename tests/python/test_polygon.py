"""test_polygon - the Polygon class in Python: constructors, queries, maps and Boolean operations.

CORA authors: Niklas Kochdumper, Tobias Ladner, Mark Wetzlinger
Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@polygon/polygon.m
# CORA: contSet/@polygon/dim.m
# CORA: contSet/@polygon/empty.m
# CORA: contSet/@polygon/display.m
# CORA: contSet/@polygon/center.m
# CORA: contSet/@polygon/interval.m
# CORA: contSet/@polygon/vertices_.m
# CORA: contSet/@polygon/supportFunc_.m
# CORA: contSet/@polygon/mtimes.m
# CORA: contSet/@polygon/times.m
# CORA: contSet/@polygon/plus.m
# CORA: contSet/@polygon/minus.m
# CORA: contSet/@polygon/or.m
# CORA: contSet/@polygon/and_.m
# CORA: contSet/@polygon/subtract.m
# CORA: contSet/@polygon/mldivide.m
# CORA: contSet/@polygon/linComb.m
# CORA: contSet/@polygon/convHull_.m
# CORA: contSet/@polygon/polytope.m
# CORA: contSet/@polygon/project.m
# CORA: contSet/@polygon/contains_.m
# CORA: contSet/@polygon/isIntersecting_.m
# CORA: contSet/@polygon/isequal.m
# CORA: contSet/@polygon/isBounded.m
# CORA: contSet/@polygon/isConvex.m
# CORA: contSet/@polygon/isFullDim.m
# CORA: contSet/@polygon/representsa_.m
# CORA: contSet/@polygon/volume_.m
# CORA: contSet/@polygon/triangulation.m
# CORA: contSet/@polygon/splitIntoConvexSets.m
# CORA: contSet/@polygon/randPoint_.m

import unittest

import matplotlib


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# the Agg backend draws without a display; it is chosen before pyplot is imported
matplotlib.use("Agg")

import cora
import matplotlib.pyplot as plt
import numpy as np


class Polygon(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_constructors_and_the_queries(self):
        L = l_shape()
        self.assertEqual(L.dim, 2)
        self.assertGreaterEqual(L.num_pieces, 2)
        self.assertEqual(L.class_name, "Polygon")
        self.assertAlmostEqual(L.volume(), 3.0)
        self.assertTrue(L.is_bounded())
        self.assertTrue(L.is_full_dim())
        self.assertFalse(L.is_convex())
        # the text, the centroid, the box and the support value
        self.assertIn("pieces", repr(L))
        np.testing.assert_allclose(L.center, [2.5 / 3, 2.5 / 3])
        np.testing.assert_allclose(L.interval().sup, [2.0, 2.0])
        self.assertAlmostEqual(float(L.support_function(np.array([1.0, 1.0]))), 3.0)
        self.assertEqual(len(L.vertices()), L.num_pieces)
        # from pieces, from a set, and the empty polygon
        square = cora.Polytope(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        self.assertEqual(cora.Polygon([square]).num_pieces, 1)
        box = cora.Interval(np.array([0.0, 0.0]), np.array([2.0, 1.0]))
        self.assertAlmostEqual(cora.Polygon(box).volume(), 2.0)
        self.assertTrue(cora.Polygon.empty().representsa("emptySet"))
        self.assertTrue(L.representsa("polygon"))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.Polygon(np.array([[0.0, 0.0, 1.0]]))
        with self.assertRaises(cora.errors.NotSupported):
            nested = [[0, 0], [4, 0], [4, 4], [0, 4], [np.nan, np.nan], [1, 1], [2, 1], [1, 2]]
            cora.Polygon(np.array(nested, dtype=float))

    def test_the_maps_and_sums(self):
        L = l_shape()
        self.assertAlmostEqual((np.array([[2.0, 1.0], [0.0, 3.0]]) @ L).volume(), 18.0)
        self.assertAlmostEqual(L.times(np.array([2.0, 0.5])).volume(), 3.0)
        self.assertAlmostEqual((unit_square() + unit_square()).volume(), 4.0)
        np.testing.assert_allclose((L + np.array([1.0, -1.0])).interval().inf, [1.0, -1.0])
        np.testing.assert_allclose((L - np.array([1.0, 0.0])).interval().inf, [-1.0, 0.0])
        # the swap, the hull and the pieces
        self.assertAlmostEqual(L.project([1, 0]).volume(), 3.0)
        self.assertAlmostEqual(L.convex_hull().volume(), 3.5)
        self.assertAlmostEqual(L.polytope().volume(), 3.5)
        self.assertGreaterEqual(len(L.split_into_convex_sets()), 2)
        self.assertEqual(unit_square().linear_combination(unit_square()).num_pieces, 1)

    def test_the_boolean_operations(self):
        L, S = l_shape(), unit_square()
        self.assertAlmostEqual(L.intersect(S).volume(), 1.0)
        self.assertAlmostEqual((L & S).volume(), 1.0)
        self.assertAlmostEqual(L.set_difference(S).volume(), 2.0)
        self.assertAlmostEqual((L | S).volume(), 3.0)
        self.assertAlmostEqual(sum(T.volume() for T in L.triangulation()), 3.0)
        # the predicates and the random points
        self.assertTrue(L.contains(np.array([0.5, 1.5])))
        self.assertFalse(L.contains(np.array([1.5, 1.5])))
        self.assertTrue(L.is_intersecting(S))
        self.assertTrue(L.isequal(L))
        X = L.rand_point(50, cora.Rng(3))
        self.assertEqual(X.shape, (50, 2))
        self.assertTrue(L.contains(X))

    def test_plot_draws_the_union(self):
        fig, ax = plt.subplots()
        cora.plot(l_shape(), ax=ax)
        self.assertGreaterEqual(len(ax.patches), 1)
        plt.close(fig)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def unit_square():
    return cora.Polygon(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))


def l_shape():
    """The L of three unit squares: [0, 2] x [0, 1] and [0, 1] x [1, 2]."""
    rows = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    return cora.Polygon(np.array(rows, dtype=float))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
