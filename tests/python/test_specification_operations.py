"""test_specification_operations - time windows, invariants, lists and robustness of specifications in Python.

CORA authors: Niklas Kochdumper, Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: specification/@specification/check.m
# CORA: specification/@specification/specification.m
# CORA: specification/@specification/add.m
# CORA: specification/@specification/isempty.m
# CORA: specification/@specification/project.m
# CORA: specification/@specification/isequal.m
# CORA: specification/@specification/eq.m
# CORA: specification/@specification/ne.m
# CORA: specification/@specification/inverse.m
# CORA: specification/@specification/robustness.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Windows(unittest.TestCase):
    def test_a_timed_specification_is_active_in_its_window(self):
        # a window, a set before it, a set in it, and no time
        spec = cora.Specification.unsafe_set(box(0, 1, 0, 1)).with_time(1.0, 2.0)
        self.assertEqual(spec.time, (1.0, 2.0))
        Z = cora.Zonotope(np.array([0.5, 0.5]), 0.1 * np.eye(2))
        self.assertTrue(spec.check(Z, (0.0, 0.5)))
        self.assertFalse(spec.check(Z, (1.5, 1.5)))
        with self.assertRaises(cora.errors.SpecialError):
            spec.check(Z)

    def test_points_with_times(self):
        spec = cora.Specification.unsafe_set(box(0, 1, 0, 1)).with_time(1.0, 2.0)
        P = np.array([[0.5, 0.5], [3.0, 3.0]])
        self.assertTrue(spec.check_points(P, [0.0, 1.5]))
        self.assertFalse(spec.check_points(P, [1.5, 0.0]))

    # an invariant and a reach set
    def test_an_invariant_must_intersect(self):
        spec = cora.Specification.invariant(box(0, 1, 0, 1))
        self.assertEqual(spec.type, "invariant")
        self.assertTrue(spec.check(cora.Zonotope(np.array([0.5, 0.5]), 0.1 * np.eye(2))))
        self.assertFalse(spec.check(cora.Zonotope(np.array([3.0, 3.0]), 0.1 * np.eye(2))))

    def test_a_reach_set_is_checked_against_the_window(self):
        # the sets of three steps and the points at their borders
        R = cora.ReachSet([cora.Zonotope(np.array([x, 0.5]), 0.1 * np.eye(2)) for x in (0.5, 1.5, 2.5)],
                          [cora.Zonotope(np.array([x, 0.5]), np.zeros((1, 2))) for x in (0.0, 1.0, 2.0, 3.0)],
                          1.0)
        band = box(1.4, 1.6, 0.0, 1.0)
        self.assertFalse(cora.Specification.unsafe_set(band).check(R))
        self.assertTrue(cora.Specification.unsafe_set(band).with_time(2.5, 2.6).check(R))
        self.assertFalse(cora.Specification.unsafe_set(band).with_time(1.2, 1.3).check(R))


class Lists(unittest.TestCase):
    def test_add_is_empty_project_and_equality(self):
        # lists, projection, equality and locations
        a = cora.Specification.safe_set(box(0, 1, 0, 1))
        u = cora.Specification.unsafe_set(box(2, 3, 2, 3))
        self.assertTrue(cora.Specification.is_empty([]))
        self.assertEqual(len(cora.Specification.add([a], [u])), 2)
        self.assertTrue(a == cora.Specification.safe_set(box(0, 1, 0, 1)))
        self.assertTrue(a != u)
        self.assertTrue(cora.Specification.isequal_lists([a, u], [u, a]))
        self.assertEqual(a.with_time(0.0, 1.0).project([1]).set.A.shape[1], 1)
        self.assertEqual(a.with_locations([0, 1]).locations, [0, 1])

    def test_the_inverse_of_safe_sets_is_their_intersection_as_an_unsafe_set(self):
        # safe sets invert to the unsafe set of their intersection
        inv = cora.Specification.inverse([cora.Specification.safe_set(box(0, 1, 0, 1)),
                                          cora.Specification.safe_set(box(0.5, 2, 0, 1))])
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0].type, "unsafe_set")
        self.assertTrue(inv[0].set.isequal(box(0.5, 1, 0, 1)))


class Robustness(unittest.TestCase):
    def test_signed_distances(self):
        # inside, one hyperplane violated, a vertex of two
        P = np.array([[0.25, 0.5], [2.0, 0.5], [2.0, 2.0]])
        safe = cora.Specification.safe_set(box(0, 1, 0, 1))
        unsafe = cora.Specification.unsafe_set(box(0, 1, 0, 1))
        np.testing.assert_allclose(cora.Specification.robustness([safe], P), [0.25, -1.0, -math.sqrt(2)])
        np.testing.assert_allclose(cora.Specification.robustness([unsafe], P), [-0.25, 1.0, math.sqrt(2)])

    def test_a_trajectory_in_a_window(self):
        # only the point in the window counts
        spec = cora.Specification.safe_set(box(0, 1, 0, 1)).with_time(1.0, 2.0)
        P = np.array([[0.5, 0.5], [2.0, 0.5], [5.0, 5.0]])
        self.assertAlmostEqual(cora.Specification.robustness([spec], P, [0.0, 1.5, 3.0]), -1.0)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def box(lo1, hi1, lo2, hi2):
    return cora.Polytope(np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]]), np.array([hi1, -lo1, hi2, -lo2]))


# run the tests
if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
