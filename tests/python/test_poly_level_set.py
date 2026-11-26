"""test_poly_level_set - the PolyLevelSet class in Python: the rows of a constrained polynomial zonotope.

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Bindings(unittest.TestCase):
    # the fields, the point test and the operations
    def test_rows(self):
        P = known()
        self.assertEqual(P.num_rows, 1)
        self.assertEqual(P.num_vars, 2)
        self.assertEqual(P.R.shape, (2, 2))
        self.assertTrue(P.contains(np.array([[0.5, -0.4]]), tol=1e-12))
        self.assertFalse(P.contains(np.array([[0.0, 0.0]]), tol=1e-12))
        self.assertEqual(P.cart_prod(P).num_vars, 4)
        self.assertEqual(P.compact().num_rows, 1)
        self.assertEqual(P.get_subset(0, 0.25, 0.5).num_vars, 2)
        self.assertIsInstance(P.lhs([1, 2]), cora.PolyZonotope)
        self.assertIsInstance(P.level_set(), cora.LevelSet)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def known():
    """The row a1 - 0.5 a2 = 0.7 on two factors."""
    return cora.PolyLevelSet(np.array([[1.0, -0.5]]), np.array([0.7]),
                             np.array([[1, 0], [0, 1]]), 2)


# ===========================================  MAIN  =========================================== #


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------  END OF CODE  ----------------------------------------- #
