"""test_fullspace - the whole space R^n: constructor, set operations, predicates, lift and linear map.

CORA authors: Mark Wetzlinger, Tobias Ladner
Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
Written:       03-November-2026
Last update:   04-November-2026
Reviewed:      ---
"""
# CORA: contSet/@fullspace/Inf.m
# CORA: contSet/@fullspace/and_.m
# CORA: contSet/@fullspace/box.m
# CORA: contSet/@fullspace/center.m
# CORA: contSet/@fullspace/contains_.m
# CORA: contSet/@fullspace/convHull_.m
# CORA: contSet/@fullspace/enclosePoints.m
# CORA: contSet/@fullspace/fullspace.m
# CORA: contSet/@fullspace/interval.m
# CORA: contSet/@fullspace/isBounded.m
# CORA: contSet/@fullspace/isFullDim.m
# CORA: contSet/@fullspace/isIntersecting_.m
# CORA: contSet/@fullspace/isequal.m
# CORA: contSet/@fullspace/dim.m
# CORA: contSet/@fullspace/display.m
# CORA: contSet/@fullspace/generateRandom.m
# CORA: contSet/@fullspace/lift_.m
# CORA: contSet/@fullspace/mtimes.m
# CORA: contSet/@fullspace/not.m
# CORA: contSet/@fullspace/plus.m
# CORA: contSet/@fullspace/project.m
# CORA: contSet/@fullspace/projectHighDim_.m
# CORA: contSet/@fullspace/radius.m
# CORA: contSet/@fullspace/randPoint_.m
# CORA: contSet/@fullspace/representsa_.m
# CORA: contSet/@fullspace/supportFunc_.m
# CORA: contSet/@fullspace/vertices_.m
# CORA: contSet/@fullspace/volume_.m
# PROPERTY: contSet/@fullspace/mtimes.m
# PROPERTY: contSet/@fullspace/representsa_.m
# PROPERTY: contSet/@fullspace/supportFunc_.m
# PROPERTY: contSet/@fullspace/vertices_.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

INF = float("inf")


def box(lo, hi):
    return cora.Interval(np.array(lo), np.array(hi))


# ============================================  MAIN  ============================================ #

class Construction(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_constructor_inf_and_box(self):
        F = cora.Fullspace(3)
        self.assertEqual(F.dim, 3)
        self.assertEqual(cora.Fullspace.inf(2).dim, 2)
        self.assertEqual(F.box().dim, 3)
        self.assertEqual(cora.Fullspace(0).dim, 0)
        self.assertIn("fullspace", repr(F))
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.Fullspace(-1)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Fullspace.inf(-1)

    def test_center_and_interval(self):
        self.assertEqual(cora.Fullspace(3).center.tolist(), [0.0, 0.0, 0.0])
        I = cora.Fullspace(2).interval()
        self.assertEqual(I.inf.tolist(), [-INF, -INF])
        self.assertEqual(I.sup.tolist(), [INF, INF])
        with self.assertRaises(cora.errors.OutOfDomain):
            # the full space of dimension 0
            cora.Fullspace(0).interval()

    def test_enclose_points_and_lift(self):
        self.assertEqual(cora.Fullspace.enclose_points(np.zeros((3, 4))).dim, 3)
        self.assertEqual(cora.Fullspace(2).lift(5, [4, 0]).dim, 5)
        with self.assertRaises(cora.errors.WrongValue):
            cora.Fullspace(2).lift(3, [0, 3])
        with self.assertRaises(cora.errors.WrongValue):
            cora.Fullspace(2).lift(1, [0, 0])
        with cora.checks(False):
            self.assertEqual(cora.Fullspace(2).lift(3, [2, 0]).dim, 3)


class SetOperations(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.F = cora.Fullspace(2)
        self.Z = cora.Zonotope(np.array([1.0, 2.0]), np.eye(2))
        self.I = box([0.0, 1.0], [2.0, 3.0])

    def test_intersect_is_the_operand(self):
        self.assertIsInstance(self.F.intersect(self.Z), cora.Zonotope)
        self.assertIsInstance(self.F & self.I, cora.Interval)
        self.assertIsInstance(cora.intersect(self.I, self.F), cora.Interval)
        # intersections
        self.assertIsInstance(self.F.intersect(cora.Fullspace(2)), cora.Fullspace)
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.F.intersect(cora.Fullspace(3))

    def test_plus_and_convex_hull_are_the_whole_space(self):
        for other in (self.Z, self.I, cora.Fullspace(2)):
            self.assertIsInstance(self.F + other, cora.Fullspace)
            self.assertIsInstance(other + self.F, cora.Fullspace)
            self.assertIsInstance(self.F.convex_hull(other), cora.Fullspace)
            self.assertIsInstance(cora.convex_hull(other, self.F), cora.Fullspace)
        self.assertIsInstance(self.F + np.array([1.0, 2.0]), cora.Fullspace)
        self.assertIsInstance(self.F.plus(np.array([1.0, 2.0])), cora.Fullspace)
        with self.assertRaises(cora.errors.NotSupported):
            cora.Fullspace(0).plus(cora.Fullspace(0))

    def test_the_sum_with_an_empty_set_is_empty(self):
        self.assertIsInstance(self.F + cora.Interval.empty(2), cora.EmptySet)
        self.assertIsInstance(self.F.plus(cora.EmptySet(2)), cora.EmptySet)
        self.assertIsInstance(cora.EmptySet(2) + self.F, cora.EmptySet)
        self.assertIsInstance(cora.Fullspace(0).plus(cora.EmptySet(0)), cora.EmptySet)
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.F.plus(cora.EmptySet(3))

    # containment
    def test_contains_and_is_intersecting(self):
        self.assertTrue(self.F.contains(self.Z))
        self.assertTrue(self.F.contains(np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(cora.contains(self.F, self.I))
        self.assertTrue(self.F.is_intersecting(self.I))
        self.assertTrue(self.F.is_intersecting(np.array([5.0, 5.0])))
        self.assertFalse(self.F.is_intersecting(cora.Interval.empty(2)))
        self.assertFalse(self.F.is_intersecting(cora.EmptySet(2)))
        self.assertTrue(cora.is_intersecting(self.Z, self.F))
        with self.assertRaises(cora.errors.NotSupported):
            cora.Fullspace(0).contains(cora.Fullspace(0))
        with self.assertRaises(cora.errors.NotSupported):
            cora.Fullspace(0).is_intersecting(cora.Fullspace(0))
        with self.assertRaises(cora.errors.WrongValue):
            self.F.contains(self.Z, tol=-1.0)

    def test_isequal(self):
        self.assertTrue(self.F.isequal(cora.Fullspace(2)))
        self.assertFalse(self.F.isequal(cora.Fullspace(3)))
        self.assertTrue(self.F.isequal(cora.Interval.unbounded(2)))
        self.assertFalse(self.F.isequal(box([-INF, 0.0], [INF, INF])))
        # equality
        self.assertFalse(self.F.isequal(cora.Interval.empty(2)))
        self.assertFalse(self.F.isequal(self.Z))
        self.assertFalse(self.F.isequal(np.array([0.0, 0.0])))
        with self.assertRaises(cora.errors.WrongValue):
            self.F.isequal(self.F, tol=-1.0)

    def test_bounded_and_full_dim(self):
        self.assertFalse(self.F.is_bounded())
        self.assertTrue(cora.Fullspace(0).is_bounded())
        self.assertTrue(self.F.is_full_dim())


class LinearMap(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))
        self.F = cora.Fullspace(2)

    def test_cases_of_cora(self):
        self.assertIsInstance(self.F.linear_map(np.array([[1.0, 2.0], [3.0, 4.0]])), cora.Fullspace)
        tall = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) @ self.F
        self.assertEqual(tall.dim, 3)
        zero_row = np.array([[1.0, 2.0], [0.0, 0.0]]) @ self.F
        self.assertEqual(zero_row.inf.tolist(), [-INF, 0.0])
        self.assertEqual(zero_row.sup.tolist(), [INF, 0.0])
        # scaling by zero
        self.assertIsInstance(2.0 * self.F, cora.Fullspace)
        self.assertEqual((0.0 * self.F).sup.tolist(), [0.0, 0.0])
        with self.assertRaises(cora.errors.DimensionMismatch):
            self.F.linear_map(np.ones((1, 3)))
        with self.assertRaises(cora.errors.NotSupported):
            cora.Fullspace(0).linear_map(np.ones((1, 1)))
        with self.assertRaises(cora.errors.Noops):
            self.F.linear_map(cora.Interval.unbounded(2))

    def test_the_result_encloses_the_image(self):
        gen = np.random.default_rng(4)
        for trial in range(20):
            M = gen.standard_normal((3, 2))
            if trial % 3 == 0:
                M[1] = 0.0
            hull = (M @ self.F).interval()
            y = M @ gen.standard_normal((2, 5))
            self.assertTrue(np.all(hull.inf[:, None] <= y) and np.all(y <= hull.sup[:, None]))


class ContSetInterface(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    # support function and projection
    def test_support_function_project_and_rand_point(self):
        F = cora.Fullspace(3)
        self.assertEqual(float(F.support_function(np.array([1.0, 0.0, 0.0]))), INF)
        self.assertEqual(float(F.support_function(np.zeros(3))), 0.0)
        self.assertEqual(F.project([0, 2]).dim, 2)
        self.assertEqual(F.rand_point(4, cora.Rng(1)).shape, (3, 4))
        with self.assertRaises(cora.errors.NotSupported):
            cora.Fullspace(0).rand_point(1, cora.Rng(1))
        self.assertEqual(len(cora.Fullspace(2).vertices()[0]), 4)
        self.assertEqual(len(cora.Fullspace(3).vertices_3d()[0]), 8)
        self.assertIn("dimension: 3", repr(F))
        with self.assertRaises(cora.errors.WrongValue):
            F.project([3])
        with self.assertRaises(cora.errors.NotSupported):
            cora.Fullspace(3).vertices()

    def test_support_function_lies_above_every_set(self):
        gen = np.random.default_rng(12)
        for n in (1, 2, 3):
            Z = cora.Zonotope(gen.standard_normal(n), gen.standard_normal((n, 3)))
            for _ in range(5):
                # the full space has the larger support
                d = gen.standard_normal(n)
                self.assertGreaterEqual(float(cora.Fullspace(n).support_function(d)),
                                        float(Z.support_function(d)))

    def test_vertices_are_the_corners_of_the_box(self):
        corners = cora.Fullspace(2).vertices()[0]
        self.assertEqual({tuple(np.sign(p)) for p in corners}, {(1, 1), (1, -1), (-1, 1), (-1, -1)})
        self.assertTrue(all(np.isinf(p).all() for p in corners))

    def test_project_high_dim_radius_volume_and_complement(self):
        F = cora.Fullspace(2)
        self.assertEqual(F.project_high_dim(2, [1, 0]).dim, 2)
        with self.assertRaises(cora.errors.NotDefined):
            F.project_high_dim(4, [0, 3])
        with self.assertRaises(cora.errors.WrongValue):
            F.project_high_dim(1, [0, 0])
        self.assertEqual((F.radius(), F.volume()), (INF, INF))
        self.assertTrue(np.isnan(cora.Fullspace(0).radius()) and np.isnan(cora.Fullspace(0).volume()))
        E = F.complement()
        self.assertIsInstance(E, cora.EmptySet)
        self.assertEqual(E.dim, 2)

    def test_representsa(self):
        # classes it represents
        F = cora.Fullspace(3)
        for type_ in ("interval", "convexSet", "fullspace"):
            self.assertTrue(F.representsa(type_), type_)
        for type_ in ("origin", "point", "zonotope", "polytope", "emptySet", "halfspace"):
            self.assertFalse(F.representsa(type_), type_)
        for type_ in ("zonotope", "emptySet", "fullspace"):
            self.assertTrue(cora.Fullspace(0).representsa(type_), type_)
        for type_ in ("origin", "point", "parallelotope"):
            self.assertFalse(cora.Fullspace(0).representsa(type_), type_)
        with self.assertRaises(cora.errors.WrongValue):
            F.representsa("circle")
        with self.assertRaises(cora.errors.NotSupported):
            F.representsa("levelSet")
        with self.assertRaises(cora.errors.NotSupported):
            F.representsa("polygon")
        with self.assertRaises(cora.errors.WrongValue):
            F.representsa("interval", tol=-1.0)

    def test_generate_random(self):
        rng = cora.Rng(5)
        self.assertEqual(cora.Fullspace.generate_random(rng, dimension=4).dim, 4)
        # random dimensions
        drawn = {cora.Fullspace.generate_random(rng).dim for _ in range(200)}
        self.assertEqual(drawn, set(range(1, 11)))
        with self.assertRaises(cora.errors.WrongValue):
            cora.Fullspace.generate_random(rng, dimension=-1)

    def test_is_full_dim_takes_a_tolerance(self):
        S = cora.Fullspace(2)
        self.assertEqual(S.is_full_dim(1e-3), S.is_full_dim())
        with self.assertRaises(cora.errors.WrongValue):
            S.is_full_dim(tol=-1.0)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
