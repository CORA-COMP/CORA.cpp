"""test_empty_set - the EmptySet class in Python: a set without points of n dimensions.

CORA authors: Mark Wetzlinger, Tobias Ladner
Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       03-November-2026
Last update:   04-November-2026
Reviewed:      ---
"""
# CORA: contSet/@emptySet/and_.m
# CORA: contSet/@emptySet/center.m
# CORA: contSet/@emptySet/contains_.m
# CORA: contSet/@emptySet/convHull_.m
# CORA: contSet/@emptySet/dim.m
# CORA: contSet/@emptySet/display.m
# CORA: contSet/@emptySet/empty.m
# CORA: contSet/@emptySet/emptySet.m
# CORA: contSet/@emptySet/interval.m
# CORA: contSet/@emptySet/isBounded.m
# CORA: contSet/@emptySet/isFullDim.m
# CORA: contSet/@emptySet/isIntersecting_.m
# CORA: contSet/@emptySet/isequal.m
# CORA: contSet/@emptySet/generateRandom.m
# CORA: contSet/@emptySet/lift_.m
# CORA: contSet/@emptySet/mtimes.m
# CORA: contSet/@emptySet/not.m
# CORA: contSet/@emptySet/plus.m
# CORA: contSet/@emptySet/project.m
# CORA: contSet/@emptySet/projectHighDim_.m
# CORA: contSet/@emptySet/radius.m
# CORA: contSet/@emptySet/randPoint_.m
# CORA: contSet/@emptySet/representsa_.m
# CORA: contSet/@emptySet/supportFunc_.m
# CORA: contSet/@emptySet/vertices_.m
# CORA: contSet/@emptySet/volume_.m
# PROPERTY: contSet/@emptySet/representsa_.m
# PROPERTY: contSet/@emptySet/supportFunc_.m

import unittest

import matplotlib

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

matplotlib.use("Agg")
import cora
import matplotlib.pyplot as plt
import numpy as np

INF = float("inf")


def box(lo, hi):
    return cora.Interval(np.array(lo), np.array(hi))


# ============================================  MAIN  ============================================ #

class EmptySet(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.E = cora.EmptySet(2)

    def test_the_constructor_and_empty(self):
        self.assertEqual(self.E.dim, 2)
        self.assertEqual(self.E.class_name, "EmptySet")
        self.assertEqual(cora.EmptySet.empty(4).dim, 4)
        self.assertEqual(cora.EmptySet.empty().dim, 0)
        self.assertIn("2-dimensional empty set", repr(self.E))
        with self.assertRaises(cora.errors.NoInputInSetConstructor):
            cora.EmptySet()
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.EmptySet(-1)
        with self.assertRaises(cora.errors.WrongValue):
            cora.EmptySet.empty(-1)

    def test_center_points_and_support_function(self):
        self.assertEqual(self.E.center.shape[-1], 0)
        self.assertEqual(self.E.rand_point(3, cora.Rng(1)).shape[-1], 0)
        self.assertEqual(self.E.support_function(np.array([1.0, 2.0])), -INF)
        self.assertTrue(self.E.interval().is_bounded())
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.E.support_function(np.array([1.0, 2.0, 3.0]))

    def test_interval_project_lift_and_vertices(self):
        self.assertEqual(self.E.interval().dim, 2)
        P = cora.EmptySet(4).project([2, 0])
        self.assertIsInstance(P, cora.EmptySet)
        self.assertEqual(P.dim, 2)
        self.assertEqual(self.E.lift(5, [0, 3]).dim, 5)
        self.assertEqual(self.E.vertices(), [])
        with self.assertRaises(cora.errors.WrongValue):
            self.E.project([2])
        with self.assertRaises(cora.errors.WrongValue):
            self.E.lift(1, [0, 0])

    def test_predicates(self):
        self.assertTrue(self.E.is_bounded())
        self.assertFalse(self.E.is_full_dim())
        Z = cora.Zonotope(np.zeros(2), np.eye(2))
        self.assertFalse(self.E.is_intersecting(Z))
        self.assertFalse(cora.is_intersecting(Z, self.E))
        self.assertTrue(self.E.isequal(cora.EmptySet(2)))
        self.assertFalse(self.E.isequal(cora.EmptySet(3)))
        self.assertTrue(self.E.isequal(cora.Interval.empty(2)))
        self.assertFalse(self.E.isequal(Z))
        with self.assertRaises(cora.errors.WrongValue):
            self.E.isequal(self.E, -1.0)

    def test_containment(self):
        Z = cora.Zonotope(np.zeros(2), np.eye(2))
        self.assertTrue(self.E.contains(cora.EmptySet(2)))
        self.assertTrue(self.E.contains(cora.Interval.empty(2)))
        self.assertFalse(self.E.contains(Z))
        self.assertTrue(cora.contains(self.E, cora.Interval.empty(2)))
        self.assertTrue(self.E.contains(np.zeros((2, 0))))
        self.assertFalse(self.E.contains(np.zeros((2, 1))))
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.E.contains(cora.EmptySet(3))

    def test_sets_combine_into_the_empty_set_or_the_other_set(self):
        Z = cora.Zonotope(np.zeros(2), np.eye(2))
        I = box([0.0, 0.0], [1.0, 1.0])
        for S in (Z, I, cora.EmptySet(2)):
            self.assertIsInstance(self.E + S, cora.EmptySet)
            self.assertIsInstance(S + self.E, cora.EmptySet)
            self.assertIsInstance(self.E & S, cora.EmptySet)
            self.assertIsInstance(S & self.E, cora.EmptySet)
            self.assertIsInstance(self.E.plus(S), cora.EmptySet)
            self.assertIsInstance(self.E.intersect(S), cora.EmptySet)
        self.assertIsInstance(self.E.plus(np.array([1.0, 2.0])), cora.EmptySet)
        self.assertIsInstance(self.E.convex_hull(Z), cora.Zonotope)
        self.assertIsInstance(cora.convex_hull(I, self.E), cora.Interval)
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.E.intersect(cora.EmptySet(3))
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.E.plus(np.ones(3))

    def test_linear_map(self):
        E = cora.EmptySet(3)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual((M @ E).dim, 2)
        self.assertEqual(E.linear_map(M).dim, 2)
        self.assertEqual((2.5 * E).dim, 3)
        self.assertEqual((cora.Interval.matrix(0.5 * M, 2 * M) @ E).dim, 2)
        with self.assertRaises(cora.errors.DimensionMismatch):
            E.linear_map(np.ones((2, 2)))

    def test_project_high_dim_radius_volume_and_complement(self):
        E = cora.EmptySet(2)
        self.assertEqual(E.project_high_dim(5, [3, 0]).dim, 5)
        with self.assertRaises(cora.errors.WrongValue):
            E.project_high_dim(1, [0, 0])
        self.assertEqual((E.radius(), E.volume()), (0.0, 0.0))
        F = E.complement()
        self.assertIsInstance(F, cora.Fullspace)
        self.assertEqual(F.dim, 2)
        self.assertIsInstance(F.complement(), cora.EmptySet)

    def test_representsa(self):
        for type_ in ("zonotope", "interval", "polytope", "emptySet", "convexSet", "capsule"):
            self.assertTrue(self.E.representsa(type_), type_)
        for type_ in ("origin", "point", "fullspace", "halfspace", "hyperplane", "parallelotope"):
            self.assertFalse(self.E.representsa(type_), type_)
        with self.assertRaises(cora.errors.WrongValue):
            self.E.representsa("circle")
        with self.assertRaises(cora.errors.WrongValue):
            self.E.representsa("zonotope", tol=-1.0)
        with self.assertRaises(cora.errors.NotSupported):
            self.E.representsa("levelSet")
        with self.assertRaises(cora.errors.NotSupported):
            self.E.representsa("polygon")

    def test_support_function_lies_below_every_set(self):
        gen = np.random.default_rng(11)
        for n in (1, 2, 3):
            Z = cora.Zonotope(gen.standard_normal(n), gen.standard_normal((n, 3)))
            for _ in range(5):
                d = gen.standard_normal(n)
                rho = float(cora.EmptySet(n).support_function(d))
                self.assertEqual(rho, -INF)
                self.assertLessEqual(rho, float(Z.support_function(d)))

    def test_generate_random(self):
        rng = cora.Rng(5)
        self.assertEqual(cora.EmptySet.generate_random(rng, dimension=4).dim, 4)
        drawn = {cora.EmptySet.generate_random(rng).dim for _ in range(200)}
        self.assertEqual(drawn, set(range(1, 11)))
        with self.assertRaises(cora.errors.WrongValue):
            cora.EmptySet.generate_random(rng, dimension=-1)

    def test_is_full_dim_takes_a_tolerance(self):
        S = cora.EmptySet(2)
        self.assertEqual(S.is_full_dim(1e-3), S.is_full_dim())
        with self.assertRaises(cora.errors.WrongValue):
            S.is_full_dim(tol=-1.0)

    def test_pairs_with_the_whole_space(self):
        F = cora.Fullspace(2)
        self.assertIsInstance(self.E + F, cora.EmptySet)
        self.assertIsInstance(F + self.E, cora.EmptySet)
        self.assertIsInstance(self.E & F, cora.EmptySet)
        self.assertIsInstance(cora.convex_hull(self.E, F), cora.Fullspace)
        self.assertFalse(cora.contains(self.E, F))
        self.assertTrue(cora.contains(F, self.E))
        self.assertFalse(cora.is_intersecting(F, self.E))
        self.assertIn(("plus", "EmptySet", "Fullspace"), set(cora.binary_pairs()))

    def test_plotting_draws_nothing(self):
        fig, ax = plt.subplots()
        cora.plot(cora.EmptySet(2), ax=ax)
        self.assertEqual([len(p.get_path().vertices) for p in ax.patches], [0])
        plt.close(fig)

    def test_the_registry_lists_the_pairs(self):
        pairs = set(cora.binary_pairs())
        self.assertIn(("plus", "EmptySet", "Zonotope"), pairs)
        self.assertIn(("contains", "EmptySet", "Interval"), pairs)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
