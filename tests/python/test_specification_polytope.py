"""test_specification_polytope - an unsafe set of several halfspaces in Python (numpy, as MATLAB CORA)."""
import unittest

import numpy as np

import cora


def box(l1, l2, u1, u2):
    return np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]]), np.array([u1, u2, -l1, -l2])


class UnsafePolytope(unittest.TestCase):
    def test_a_zonotope_is_checked_against_the_intersection_of_the_halfspaces(self):
        Z = cora.Zonotope(np.array([2.0, 1.0]), np.eye(2))  # [1, 3] x [0, 2]
        self.assertTrue(cora.Specification.unsafeSet(*box(0, 0, 0.5, 2)).check(Z))
        self.assertFalse(cora.Specification.unsafeSet(*box(2.5, 0.5, 5, 1)).check(Z))
        self.assertFalse(cora.Specification.unsafeSet(*box(3, 0, 5, 2)).check(Z))  # touching

    def test_no_single_halfspace_separates_the_segment(self):
        Z = cora.Zonotope(np.array([1.0, 1.0]), np.array([[1.0], [1.0]]))
        A = np.array([[-1.0, 0], [0, 1]])
        self.assertTrue(cora.Specification.unsafeSet(A, np.array([-1.5, 0.5])).check(Z))
        self.assertFalse(cora.Specification.unsafeSet(A, np.array([-0.5, 1.5])).check(Z))

    def test_the_halfspaces_are_kept(self):
        spec = cora.Specification.unsafeSet(*box(0, 0, 1, 1))
        self.assertEqual(spec.type, "unsafeSet")
        self.assertEqual(len(spec.halfspaces), 4)


if __name__ == "__main__":
    unittest.main()
