"""test_polytope_maps - the Polytope operations of the second batch in Python

CORA authors: Mark Wetzlinger, Viktor Kotsev, Niklas Kochdumper, Tobias Ladner, Adrian Kulmburg,
              Matthias Althoff, Victor Gassmann
Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@emptySet/polytope.m
# CORA: contSet/@fullspace/polytope.m
# CORA: contSet/@polytope/isBounded.m
# CORA: contSet/@polytope/isFullDim.m
# CORA: contSet/@polytope/isIntersecting_.m
# CORA: contSet/@polytope/isequal.m
# CORA: contSet/@polytope/le.m
# CORA: contSet/@polytope/lift_.m
# CORA: contSet/@polytope/minkDiff.m
# CORA: contSet/@polytope/minus.m
# CORA: contSet/@polytope/mldivide.m
# CORA: contSet/@polytope/mtimes.m
# CORA: contSet/@polytope/normalizeConstraints.m
# CORA: contSet/@polytope/not.m
# CORA: contSet/@polytope/or.m
# CORA: contSet/@polytope/origin.m
# CORA: contSet/@polytope/plus.m
# CORA: contSet/@polytope/project.m
# CORA: contSet/@polytope/projectHighDim.m
# CORA: contSet/@polytope/reduceOverDomain.m
# CORA: contSet/@polytope/representsa_.m
# CORA: contSet/@polytope/supportFunc_.m
# CORA: contSet/@polytope/volume_.m
# CORA: contSet/@polytope/zonotope.m
# CORA: contSet/@polytope/zonoBundle.m
# CORA: contSet/@polytope/ellipsoid.m
# CORA: contSet/@polytope/dim.m
# CORA: contSet/@polytope/distance.m
# CORA: contSet/@polytope/generateRandom.m
# CORA: contSet/@polytope/randPoint_.m
# CORA: contSet/@polytope/reduce.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

SQUARE_H = (np.vstack([np.eye(2), -np.eye(2)]), np.array([1.0, 1.0, 0.0, 0.0]))


def square():
    return cora.Polytope(*SQUARE_H)


def column(*x):
    return np.array(x, dtype=float).reshape(-1, 1)


# ============================================  MAIN  ============================================ #


class PolytopeMaps(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    # the origin represents itself, the square is a box
    def test_origin_and_representsa(self):
        P = cora.Polytope.origin(3)
        self.assertTrue(P.representsa("origin"))
        self.assertFalse(square().representsa("origin"))
        self.assertTrue(square().representsa("interval"))

    # the volume and the dimension queries of the unit square
    def test_volume_and_dimension_queries(self):
        P = square()
        self.assertAlmostEqual(P.volume(), 1.0, places=9)
        self.assertTrue(P.is_bounded())
        self.assertTrue(P.is_full_dim())
        full, _ = P.is_full_dim_subspace()
        self.assertTrue(full)

    # the operators shift, add and map
    def test_the_sum_shifts_and_adds(self):
        P = square()
        shifted = P + column(2.0, 0.0)
        self.assertTrue(shifted.contains(column(2.5, 0.5)))
        self.assertFalse(shifted.contains(column(0.5, 0.5)))
        back = shifted - column(2.0, 0.0)
        self.assertTrue(back.isequal(P))
        both = P + P
        self.assertAlmostEqual(both.volume(), 4.0, places=8)
        self.assertTrue((column(1.0, 1.0) + P).contains(column(1.5, 1.5)))

    # a matrix scales the area
    def test_the_linear_map_scales(self):
        P = (2.0 * np.eye(2)) * square()
        self.assertAlmostEqual(P.volume(), 4.0, places=8)

    # a projection keeps the chosen coordinates; the high-dimensional set pads zeros
    def test_projection_and_lifting(self):
        P = square().project([0])
        self.assertTrue(P.contains(column(0.5)))
        self.assertFalse(P.contains(column(1.5)))
        high = square().project_high_dim(3, [0, 2])
        self.assertEqual(high.dim, 3)
        self.assertTrue(high.contains(column(0.5, 0.0, 0.5)))
        self.assertFalse(high.contains(column(0.5, 1.0, 0.5)))

    # the intersection test and the equality, the subset operator
    def test_the_intersection_test(self):
        P = square()
        self.assertTrue(P.is_intersecting(P + column(0.5, 0.5)))
        self.assertFalse(P.is_intersecting(P + column(3.0, 0.0)))
        self.assertTrue(P.is_intersecting(column(0.5, 0.5)))
        self.assertTrue(P.isequal(square()))
        self.assertTrue(square() <= (square() + column(0.0, 0.0)))

    # the distance of disjoint squares and the hull of their union
    def test_distance_and_union(self):
        P = square()
        self.assertAlmostEqual(P.distance(P + column(3.0, 0.0)), 2.0, places=8)
        U = P.unite(P + column(3.0, 0.0))
        self.assertAlmostEqual(U.volume(), 4.0, places=8)

    # random sets and their samples satisfy the constraints; the zonotope contains them
    def test_random_sets_satisfy_their_constraints(self):
        rng = cora.Rng(4)
        P = cora.Polytope.generate_random(rng, dimension=2, is_bounded=True)
        self.assertEqual(P.dim, 2)
        points = P.rand_point(5, rng)
        for k in range(5):
            self.assertTrue(P.contains(points[:, [k]], tol=1e-9))
        self.assertTrue(P.zonotope().contains(points))

    # the pieces of the difference cover what lies outside the subtrahend
    def test_the_difference_pieces_cover_it(self):
        P = square()
        parts = P.set_difference(cora.Polytope(np.vstack([np.eye(2), -np.eye(2)]),
                                               np.array([0.5, 1.0, 0.0, 0.0])))
        self.assertGreaterEqual(len(parts), 1)
        self.assertTrue(any(p.contains(column(0.75, 0.5)) for p in parts))

    # the special sets convert to polytopes
    def test_empty_and_fullspace_convert(self):
        self.assertTrue(cora.Polytope(cora.EmptySet(2)).representsa("emptySet"))
        self.assertTrue(cora.Polytope(cora.Fullspace(2)).representsa("fullspace"))


# run the tests
if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
