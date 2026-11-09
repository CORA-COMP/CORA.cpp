"""test_specification_polytope - an unsafe set of several halfspaces in Python (numpy, as MATLAB CORA).

CORA authors: Niklas Kochdumper, Tobias Ladner, Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       30-September-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: specification/@specification/check.m
# CORA: specification/@specification/specification.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def box(l1, l2, u1, u2):
    """The box as a polytope of four rows."""
    return cora.Polytope(np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]]), np.array([u1, u2, -l1, -l2]))


# ============================================  MAIN  ============================================ #

class UnsafePolytope(unittest.TestCase):
    def test_a_zonotope_is_checked_against_the_intersection_of_the_halfspaces(self):
        Z = cora.Zonotope(np.array([2.0, 1.0]), np.eye(2))  # [1, 3] x [0, 2]
        self.assertTrue(cora.Specification.unsafe_set(box(0, 0, 0.5, 2)).check(Z))
        self.assertFalse(cora.Specification.unsafe_set(box(2.5, 0.5, 5, 1)).check(Z))
        self.assertFalse(cora.Specification.unsafe_set(box(3, 0, 5, 2)).check(Z))  # touching

    def test_no_single_halfspace_separates_the_segment(self):
        Z = cora.Zonotope(np.array([1.0, 1.0]), np.array([[1.0], [1.0]]))
        A = np.array([[-1.0, 0], [0, 1]])
        self.assertTrue(cora.Specification.unsafe_set(cora.Polytope(A, np.array([-1.5, 0.5]))).check(Z))
        self.assertFalse(cora.Specification.unsafe_set(cora.Polytope(A, np.array([-0.5, 1.5]))).check(Z))

    def test_the_polytope_is_kept(self):
        spec = cora.Specification.unsafe_set(box(0, 0, 1, 1))
        self.assertEqual(spec.type, "unsafe_set")
        self.assertEqual(spec.set.A.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
