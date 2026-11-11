"""test_level_set - the LevelSet class in Python: constraints, queries, contractor and operations.

CORA authors: Niklas Kochdumper, Mark Wetzlinger
Authors:       Claude Sonnet 5.5
Written:       11-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: contSet/@levelSet/levelSet.m
# CORA: contSet/@levelSet/dim.m
# CORA: contSet/@levelSet/empty.m
# CORA: contSet/@levelSet/Inf.m
# CORA: contSet/@levelSet/contains_.m
# CORA: contSet/@levelSet/isIntersecting_.m
# CORA: contSet/@levelSet/tightenDomain.m
# CORA: contSet/@levelSet/not.m
# CORA: contSet/@levelSet/lift_.m
# CORA: contSet/@levelSet/splitEquality.m
# CORA: contSet/@levelSet/display.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def circle(r=1.0):
    """The constraint x^2 + y^2 - r^2 of two states."""
    x, y = cora.Expr.var(0), cora.Expr.var(1)
    return x ** 2 + y ** 2 - r * r


def box(lo, hi):
    return cora.Interval(np.array(lo, dtype=float), np.array(hi, dtype=float))


# ============================================  MAIN  ============================================ #

class Constraints(unittest.TestCase):
    # the dimension, the values and the derivative
    def test_values(self):
        ls = cora.LevelSet(circle(), 2, "==")
        self.assertEqual(ls.dim, 2)
        self.assertEqual(ls.num_equations, 1)
        self.assertEqual(ls.comp_ops, ["=="])
        points = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(ls.fun_han(points), [[-1.0, 0.0, 3.0]])
        self.assertAlmostEqual(ls.gradient(0, 0).eval([3.0, 4.0]), 6.0)
        F = ls.fun_han(box([0, 0], [1, 1]))
        np.testing.assert_allclose([F.inf[0], F.sup[0]], [-1.0, 1.0])

    # the empty set and the whole space
    def test_empty_and_inf(self):
        pts = np.array([[0.0], [0.0]])
        np.testing.assert_allclose(cora.LevelSet.empty(2).fun_han(pts), [[1.0]])
        np.testing.assert_allclose(cora.LevelSet.inf(2).fun_han(pts), [[-1.0]])

    # a wrong comparison or a state beyond the dimension
    def test_errors(self):
        with self.assertRaises(cora.errors.WrongValue):
            cora.LevelSet(circle(), 2, "!=")
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.LevelSet(cora.Expr.var(3), 2, "<=")


class Queries(unittest.TestCase):
    # points, boxes, and the box of another set
    def test_contains_and_intersecting(self):
        disc = cora.LevelSet(circle(), 2, "<=")
        self.assertTrue(disc.contains(np.array([0.5, 0.0])))
        self.assertFalse(disc.contains(np.array([[2.0], [0.0]])))
        self.assertTrue(disc.contains(box([-0.5, -0.5], [0.5, 0.5])))
        self.assertFalse(disc.contains(box([0, 0], [1, 1])))
        circ = cora.LevelSet(circle(), 2, "==")
        self.assertTrue(circ.is_intersecting(box([0, 0], [2, 2])))
        self.assertFalse(circ.is_intersecting(cora.Zonotope(np.array([3.0, 3.0]), 0.5 * np.eye(2))))


class Operations(unittest.TestCase):
    # the contractor cuts to the solutions and reports an empty box as None
    def test_tighten_domain(self):
        line = cora.LevelSet(cora.Expr.var(0) - 1.0, 1, "==")
        J = line.tighten_domain(box([0], [3]))
        np.testing.assert_allclose([J.inf[0], J.sup[0]], [1.0, 1.0])
        sq = cora.LevelSet(cora.Expr.var(0) ** 2 - 4.0, 1, "==")
        self.assertIsNone(sq.tighten_domain(box([-1], [1])))
        with self.assertRaises(cora.errors.NotSupported):
            line.tighten_domain(box([0], [3]), "taylor")

    # complement, split, lift and the missing operations
    def test_complement_split_lift(self):
        disc = cora.LevelSet(circle(), 2, "<=")
        self.assertEqual(disc.complement().comp_ops, ["<"])
        eq, ineq = cora.LevelSet([cora.Expr.var(0), cora.Expr.var(0) * 2.0], 1, ["==", "<="]).split_equality()
        self.assertEqual(eq.comp_ops, ["=="])
        self.assertEqual(ineq.comp_ops, ["<="])
        lifted = cora.LevelSet(circle(), 2, "==").lift(3, [2, 0])
        self.assertEqual(lifted.dim, 3)
        np.testing.assert_allclose(lifted.fun_han(np.array([[0.6], [7.0], [0.8]])), [[0.0]], atol=1e-12)
        with self.assertRaises(cora.errors.Noops):
            disc.interval()
        self.assertIn("levelSet", disc.display())


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------  END OF CODE  ----------------------------------------- #
