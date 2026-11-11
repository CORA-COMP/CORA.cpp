"""test_taylm - the Taylm class in Python: models of boxes and zonotopes, arithmetic and bounds.

CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@taylm/taylm.m
# CORA: contSet/@taylm/dim.m
# CORA: contSet/@taylm/plus.m
# CORA: contSet/@taylm/minus.m
# CORA: contSet/@taylm/uminus.m
# CORA: contSet/@taylm/uplus.m
# CORA: contSet/@taylm/times.m
# CORA: contSet/@taylm/mtimes.m
# CORA: contSet/@taylm/power.m
# CORA: contSet/@taylm/mpower.m
# CORA: contSet/@taylm/interval.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def models():
    """The models of the box [1, 3] x [0, 2]."""
    return cora.Taylm.from_interval(cora.Interval(np.array([1.0, 0.0]), np.array([3.0, 2.0])))


# ============================================  MAIN  ============================================ #

class Models(unittest.TestCase):
    # a box gives its own bounds, a zonotope its coordinates
    def test_factories(self):
        x = models()
        self.assertEqual(x[0].num_vars, 2)
        self.assertEqual(x[0].bounds(), (1.0, 3.0))
        z = cora.Taylm.from_zonotope(cora.Zonotope(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [1.0, 2.0]])))
        self.assertEqual(z[1].bounds(), (-3.0, 3.0))
        self.assertEqual(cora.Taylm.constant(2.5, 3).bounds(), (2.5, 2.5))
        self.assertEqual(cora.Taylm.from_range(-1.0, 3.0, 1).bounds(), (-1.0, 3.0))

    # sums keep the dependency, products and powers enclose
    def test_arithmetic(self):
        x = models()
        self.assertEqual((x[0] - x[0]).bounds(), (0.0, 0.0))
        self.assertEqual((x[0] + x[1]).bounds(), (1.0, 5.0))
        self.assertEqual(x[0].plus(1.0).bounds(), (2.0, 4.0))
        self.assertEqual((-x[0]).bounds(), (-3.0, -1.0))
        sq = x[0] * x[0]
        self.assertGreaterEqual(sq.bounds()[1], 9.0 - 1e-12)
        self.assertLessEqual(sq.bounds()[0], 1.0 + 1e-12)
        self.assertEqual((x[0] ** 0).bounds(), (1.0, 1.0))
        self.assertEqual(x[0].interval().inf[0], 1.0)

    # an expression on models, and the errors
    def test_evaluate_and_errors(self):
        x, y = cora.Expr.var(0), cora.Expr.var(1)
        g = cora.Taylm.evaluate((x - 1.0) ** 2 + y ** 2, models())
        self.assertGreaterEqual(g.bounds()[1], 8.0 - 1e-12)
        with self.assertRaises(cora.errors.WrongValue):
            models()[0].power(-1)
        with self.assertRaises(cora.errors.DimensionMismatch):
            models()[0].plus(cora.Taylm.constant(1.0, 5))


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------  END OF CODE  ----------------------------------------- #
