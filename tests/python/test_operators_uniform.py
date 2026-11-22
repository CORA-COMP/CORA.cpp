"""test_operators_uniform - the Python operators that every set class answers alike.

A number or a point shifts a set (+, -, and v - S), A <= B is containment, a Taylm takes numbers,
numpy defers to the operators, and the predicates of a batch of boxes answer a mask.

CORA authors: Tobias Ladner, Mark Wetzlinger
Authors:       Claude Opus 5.5
Written:       22-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@contSet/plus.m
# CORA: contSet/@contSet/minus.m
# CORA: contSet/@contSet/uminus.m
# CORA: contSet/@contSet/contains_.m
# CORA: contSet/@taylm/plus.m
# CORA: contSet/@taylm/minus.m
# CORA: contSet/@taylm/times.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Shifts(unittest.TestCase):
    def setUp(self):
        # the Eigen backend makes the results independent of an installed torch
        self.enterContext(cora.backend("eigen"))

    # a number is added to every coordinate, as NumPy broadcasts a scalar
    def test_a_number_shifts_every_vector_class(self):
        for S in vector_sets():
            with self.subTest(S.class_name):
                by_number = S + 2.0
                by_point = S + np.full(S.dim, 2.0)
                self.assertTrue(same(by_number, by_point))
                self.assertTrue(same(2.0 + S, by_point))
                self.assertTrue(same(S - 2.0, S + np.full(S.dim, -2.0)))

    # v - S is -S + v, for a point and for a number
    def test_a_point_minus_a_set(self):
        v = np.array([1.0, 2.0])
        for S in vector_sets():
            with self.subTest(S.class_name):
                self.assertTrue(same(v - S, (-S) + v))
                self.assertTrue(same(3.0 - S, (-S) + 3.0))

    # an array or a set that is no point is refused, not misread
    def test_what_is_no_shift(self):
        for S in vector_sets():
            with self.subTest(S.class_name):
                with self.assertRaises(TypeError):
                    S + "x"
                with self.assertRaises(TypeError):
                    "x" - S

    def test_an_interval_takes_numbers(self):
        I = cora.Interval(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        self.assertTrue((I - 1.0) == cora.Interval(np.array([-1.0, 0.0]), np.array([1.0, 2.0])))
        self.assertTrue((1.0 - I) == cora.Interval(np.array([-1.0, -2.0]), np.array([1.0, 0.0])))


class Comparison(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    # A <= B is B.contains(A), A >= B is A.contains(B), for the classes with a contains
    def test_subset_by_less_equal(self):
        small = cora.Zonotope(np.zeros(2), 0.5 * np.eye(2))
        big = cora.Zonotope(np.zeros(2), 2.0 * np.eye(2))
        self.assertTrue(small <= big)
        self.assertFalse(big <= small)
        self.assertTrue(big >= small)
        # a class with its own <= (the interval) keeps it
        self.assertTrue(cora.Interval(np.zeros(2), np.ones(2)) <= cora.Interval(-np.ones(2), 2 * np.ones(2)))
        with self.assertRaises(TypeError):
            _ = small <= 3

    # the empty set and the whole space answer `in`
    def test_in_for_the_trivial_sets(self):
        p = np.array([1.0, 2.0])
        self.assertTrue(p in cora.Fullspace(2))
        self.assertFalse(p in cora.EmptySet(2))


class Taylm(unittest.TestCase):
    # a number from either side, as for the other classes
    def test_numbers_from_both_sides(self):
        x = cora.Taylm.from_range(1.0, 3.0, 1)
        self.assertEqual((x + 1.0).bounds(), (2.0, 4.0))
        self.assertEqual((1.0 + x).bounds(), (2.0, 4.0))
        self.assertEqual((x - 1.0).bounds(), (0.0, 2.0))
        self.assertEqual((1.0 - x).bounds(), (-2.0, 0.0))
        self.assertEqual((2.0 * x).bounds(), (2.0, 6.0))
        self.assertEqual((x * 2.0).bounds(), (2.0, 6.0))

    # numpy defers to the operators instead of building an object array
    def test_numpy_defers(self):
        x = cora.Taylm.from_range(1.0, 3.0, 1)
        self.assertEqual((np.float64(2.0) * x).bounds(), (2.0, 6.0))
        L = level_set()
        with self.assertRaises(TypeError):
            np.eye(2) @ L


class Masks(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    # a batch of boxes answers one flag per member, one box a plain bool
    def test_the_predicates_of_a_batch_of_intervals(self):
        I = cora.Interval(np.zeros((2, 2)), np.array([[1.0, 1.0], [2.0, 2.0]]))
        p = np.array([1.5, 1.5])
        for mask, expected in [
            (I.contains(p), [False, True]),
            (I.is_intersecting(p), [False, True]),
            (I.is_intersecting(cora.Interval(np.array([1.5, 1.5]), np.array([3.0, 3.0]))), [False, True]),
            (I.isequal(I), [True, True]),
            (I.is_full_dim(), [True, True]),
            (I.is_bounded(), [True, True]),
        ]:
            self.assertEqual(mask.dtype, np.bool_)
            np.testing.assert_array_equal(mask, expected)
        one = cora.Interval(np.zeros(2), np.ones(2))
        self.assertIs(one.contains(np.array([0.5, 0.5])), True)
        self.assertIs(one.is_bounded(), True)
        # `in` is the whole batch
        self.assertFalse(p in I)


class RandPoint(unittest.TestCase):
    # every class takes the sampler type; the ones with one sampler answer 'standard' only
    def test_the_type_argument(self):
        I = cora.Interval(np.zeros(2), np.ones(2))
        self.assertEqual(I.rand_point(3, cora.Rng(1), "standard").shape, (3, 2))
        with self.assertRaises(cora.errors.NotSupported):
            I.rand_point(3, cora.Rng(1), "extreme")


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def vector_sets():
    """One set of every class whose shift is a vector, in two dimensions."""
    Z = cora.Zonotope(np.zeros(2), np.eye(2))
    return [
        Z,
        cora.ConZonotope(np.zeros(2), np.eye(2)),
        cora.ZonoBundle(Z),
        cora.Polytope(np.vstack([np.eye(2), -np.eye(2)]), np.ones(4)),
        cora.Capsule(np.zeros(2), np.array([1.0, 0.0]), 0.5),
        cora.Ellipsoid(np.zeros(2), np.eye(2)),
        cora.PolyZonotope(np.zeros(2), np.eye(2)),
    ]


def same(A, B):
    """Whether two sets of one class agree: a polynomial zonotope by its data, else by =="""
    if isinstance(A, cora.PolyZonotope):
        return np.allclose(A.c, B.c) and np.allclose(A.G, B.G) and np.allclose(A.GI, B.GI)
    return bool(A == B)


def level_set():
    """The unit disk as a level set."""
    x, y = cora.Expr.var(0), cora.Expr.var(1)
    return cora.LevelSet(x * x + y * y - 1.0, 2, "<=")


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
