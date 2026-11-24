"""test_stl_interval - the time intervals of signal temporal logic in Python.

CORA authors: Florian Lercher
Authors:       Claude Sonnet 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: specification/@stlInterval/stlInterval.m
# CORA: specification/@stlInterval/and_.m
# CORA: specification/@stlInterval/plus.m
# CORA: specification/@stlInterval/minus.m
# CORA: specification/@stlInterval/minkDiff.m
# CORA: specification/@stlInterval/leftClosure.m
# CORA: specification/@stlInterval/rightClosure.m
# CORA: specification/@stlInterval/toLeft.m
# CORA: specification/@stlInterval/toRight.m
# CORA: specification/@stlInterval/infimum.m
# CORA: specification/@stlInterval/supremum.m
# CORA: specification/@stlInterval/dim.m
# CORA: specification/@stlInterval/empty.m
# CORA: specification/@stlInterval/interval.m
# CORA: specification/@stlInterval/contains_.m
# CORA: specification/@stlInterval/isequal.m
# CORA: specification/@stlInterval/representsa_.m
# CORA: specification/@stlInterval/display.m
# CORA: specification/@stlInterval/isemptyobject.m

import math
import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Construction(unittest.TestCase):
    def test_the_forms_and_normalizations(self):
        # point, open, half open, empty by order, infinite bound open
        p = cora.StlInterval(1)
        self.assertEqual((p.lower, p.upper, p.left_closed, p.right_closed), (1, 1, True, True))
        self.assertFalse(cora.StlInterval(0, 1, False).left_closed)
        self.assertTrue(cora.StlInterval(2, 1).is_empty_object)
        self.assertTrue(cora.StlInterval(1, 1, False, True).is_empty_object)
        self.assertFalse(cora.StlInterval(0, math.inf).right_closed)
        self.assertTrue(cora.StlInterval.empty().is_empty_object)
        self.assertEqual(cora.StlInterval().dim(), 1)
        self.assertEqual(repr(cora.StlInterval(0, 2, True, False)), "StlInterval[0, 2)")

    def test_errors(self):
        # a negative lower bound and a dimension other than 1
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.StlInterval(-1, 1)
        with self.assertRaises(cora.errors.WrongValue):
            cora.StlInterval.empty(2)

    def test_endpoints_and_interval(self):
        # the bounds with their ends, and the closed Interval of the bounds
        I = cora.StlInterval(1, 2, False, True)
        self.assertEqual(I.infimum(), (1, False))
        self.assertEqual(I.supremum(), (2, True))
        J = I.interval()
        self.assertEqual((float(J.inf[0]), float(J.sup[0])), (1.0, 2.0))


class Operations(unittest.TestCase):
    def test_and_plus_minus(self):
        # intersection, sums, differences with a number and an Interval
        self.assertEqual(cora.StlInterval(0, 2, True, False).and_(cora.StlInterval(1, 3, False, True)),
                         cora.StlInterval(1, 2, False, False))
        self.assertEqual(cora.StlInterval(1, 2, False, True).plus(cora.StlInterval(2, 3)),
                         cora.StlInterval(3, 5, False, True))
        self.assertEqual(cora.StlInterval(1, 2).plus(1.5), cora.StlInterval(2.5, 3.5))
        J = cora.Interval(np.array([1.0]), np.array([2.0]))
        self.assertEqual(cora.StlInterval(1, 4, False, True).plus(J), cora.StlInterval(2, 6, False, True))
        self.assertEqual(cora.StlInterval(1, 4, False, True).minus(J), cora.StlInterval(0, 3))
        self.assertEqual(cora.StlInterval(2, 5).minus(3.0), cora.StlInterval(0, 2))
        self.assertEqual(cora.StlInterval(2, 5).minus(cora.StlInterval(1, 2)), cora.StlInterval(0, 4))

    def test_minkowski_difference_and_closures(self):
        self.assertEqual(cora.StlInterval(1, 5).minkowski_difference(cora.StlInterval(0, 2)),
                         cora.StlInterval(1, 3))
        inf = cora.StlInterval(0, math.inf)
        self.assertEqual(inf.minkowski_difference(inf), inf)
        # closures, to_left, to_right
        self.assertEqual(cora.StlInterval(1, 2, False, False).left_closure(),
                         cora.StlInterval(1, 2, True, False))
        self.assertEqual(cora.StlInterval(1, 2, False, False).right_closure(),
                         cora.StlInterval(1, 2, False, True))
        self.assertEqual(cora.StlInterval(1, 2).to_left(), cora.StlInterval(0, 1, True, False))
        self.assertEqual(cora.StlInterval(1, 2).to_right(), cora.StlInterval(2, math.inf, False, False))
        with self.assertRaises(cora.errors.EmptySet):
            cora.StlInterval().to_left()

    def test_predicates(self):
        # contains, isequal, representsa
        I = cora.StlInterval(1, 3, False, True)
        self.assertTrue(I.contains(3.0))
        self.assertFalse(I.contains(1.0))
        self.assertTrue(I.contains(cora.StlInterval(1.5, 3)))
        self.assertTrue(I.isequal(cora.StlInterval(1, 3, False, True)))
        self.assertFalse(I == cora.StlInterval(1, 3))
        self.assertTrue(cora.StlInterval(2, 2).representsa("point"))
        self.assertTrue(cora.StlInterval(0, 0).representsa("origin"))
        with self.assertRaises(cora.errors.NotSupported):
            I.representsa("zonotope")


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
