"""test_cpp_parity - what C++ offers on a set, Python offers under the same name.

The shape and the corners of a zonotope, the representation of a polytope, and the parts of a
polynomial zonotope; the older CORA names of the polynomial zonotope stay as aliases.

CORA.cpp only: CORA has no batch shape and names the parts of a polynomial zonotope c, G, GI, E, id
Authors:       Claude Opus 5.5
Written:       22-November-2026
Last update:   ---
Reviewed:      ---
"""

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Parity(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_the_shape_and_the_corners_of_a_zonotope(self):
        Z = cora.Zonotope(np.zeros(2), np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
        self.assertEqual(tuple(Z.batch_shape), ())
        self.assertEqual(tuple(Z.elem_shape), (2,))
        self.assertEqual(Z.nr_generators, 3)
        self.assertEqual(np.asarray(Z.corners()).shape, (8, 2))

    def test_the_representation_of_a_polytope(self):
        P = cora.Polytope(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([1.0, 1.0]))
        self.assertTrue(P.is_h_rep())
        self.assertFalse(P.is_v_rep())

    def test_the_parts_of_a_polynomial_zonotope(self):
        pZ = cora.PolyZonotope(cora.Zonotope(np.zeros(2), np.eye(2)))
        self.assertEqual(pZ.num_dependent, 2)
        self.assertEqual(pZ.num_factors, 2)
        np.testing.assert_array_equal(pZ.center, pZ.c)
        np.testing.assert_array_equal(pZ.generators, pZ.G)
        np.testing.assert_array_equal(pZ.independent_generators, pZ.GI)
        np.testing.assert_array_equal(pZ.exponents, pZ.E)
        self.assertEqual(list(pZ.ids), list(pZ.id))
        self.assertEqual(pZ.compact_gens().num_dependent, 2)
        self.assertEqual(cora.PolyZonotope.origin(3).center.shape, (3,))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
