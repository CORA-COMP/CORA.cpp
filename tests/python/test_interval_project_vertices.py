"""test_interval_project_vertices - the projection and the vertices of a box in Python.

CORA authors: Matthias Althoff, Mark Wetzlinger, Tobias Ladner
Authors:       Claude Sonnet 5.5
Written:       13-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@interval/project.m
# CORA: contSet/@interval/vertices_.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def box(lo, hi):
    return cora.Interval(np.array(lo, dtype=float), np.array(hi, dtype=float))


# ============================================  MAIN  ============================================ #

class ProjectVertices(unittest.TestCase):
    def setUp(self):
        self.enterContext(cora.backend("eigen"))

    def test_project_keeps_the_chosen_dimensions_in_order(self):
        P = box([1.0, 2.0, 3.0], [2.0, 4.0, 5.0]).project([2, 0])
        self.assertEqual(P.inf.tolist(), [3.0, 1.0])
        self.assertEqual(P.sup.tolist(), [5.0, 2.0])

    def test_vertices_of_a_box_are_its_four_corners(self):
        V = box([0.0, 1.0], [2.0, 3.0]).vertices()
        self.assertEqual(len(V), 1)
        corners = sorted(map(tuple, V[0].tolist()))
        self.assertEqual(corners, [(0.0, 1.0), (0.0, 3.0), (2.0, 1.0), (2.0, 3.0)])


if __name__ == "__main__":
    # run the tests
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
