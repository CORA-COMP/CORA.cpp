"""test_stl - signal temporal logic formulas and expressions in Python.

CORA authors: Florian Lercher
Authors:       Claude Sonnet 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
# CORA: specification/@stl/stl.m
# CORA: specification/@stl/and.m
# CORA: specification/@stl/or.m
# CORA: specification/@stl/not.m
# CORA: specification/@stl/implies.m
# CORA: specification/@stl/next.m
# CORA: specification/@stl/finally.m
# CORA: specification/@stl/globally.m
# CORA: specification/@stl/until.m
# CORA: specification/@stl/release.m
# CORA: specification/@stl/lt.m
# CORA: specification/@stl/le.m
# CORA: specification/@stl/gt.m
# CORA: specification/@stl/ge.m
# CORA: specification/@stl/plus.m
# CORA: specification/@stl/minus.m
# CORA: specification/@stl/times.m
# CORA: specification/@stl/power.m
# CORA: specification/@stl/isequal.m
# CORA: specification/@stl/getVariables.m
# CORA: specification/@stl/getTimes.m
# CORA: specification/@stl/maximumTime.m
# CORA: specification/@stl/minimumTime.m
# CORA: specification/@stl/desugar.m
# CORA: specification/@stl/negationNormalForm.m
# CORA: specification/@stl/eliminateTrueFalse.m
# CORA: specification/@stl/conjunctiveNormalForm.m
# CORA: specification/@stl/disjunctiveNormalForm.m
# CORA: specification/@stl/assignIdentifiers.m
# CORA: specification/@stl/combineNext.m
# CORA: specification/@stl/getClauses.m
# CORA: specification/@stl/isPredicate.m
# CORA: specification/@stl/isLinearPredicate.m
# CORA: specification/@stl/disp.m
# CORA: specification/@stl/modelCheckTrace.m
# CORA: specification/@stl/robustness.m

import unittest

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

W = cora.StlInterval(0, 2)


class Construction(unittest.TestCase):
    def setUp(self):
        self.x = cora.StlExpr.variables("x", 2)

    def test_expressions(self):
        # the operators build sums, products and powers; a number on the left works too
        x = self.x
        self.assertEqual(str(x[0] + 2), "x1 + 2")
        self.assertEqual(str(2 * x[0] - x[1]), "2*x1 - x2")
        self.assertEqual(str(1 - x[0]), "1 - x1")
        self.assertEqual(str((x[0] + x[1]) ** 2), "(x1 + x2)^2")
        self.assertTrue((x[0] + x[1]).isequal(x[1] + x[0]))
        self.assertTrue((2 * x[0] + 1).is_linear() and not (x[0] * x[1]).is_linear())
        self.assertEqual((x[0] + x[1]).variable_names(), ["x1", "x2"])
        with self.assertRaises(cora.errors.NotSupported):
            x[0] ** 0.5
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            cora.StlExpr.variables("", 1)

    def test_formulas(self):
        # comparisons, the logic operators and the temporal operators
        x = self.x
        p, q = x[0] < 5, x[1] >= 0
        self.assertEqual(str(p), "x1 < 5")
        self.assertEqual(str(5 < x[0]), "x1 > 5")
        self.assertEqual(str(x[0] <= x[1]), "x1 - x2 <= 0")
        # the logic operators
        self.assertEqual(str(p & q), "((x1 < 5) & (x2 >= 0))")
        self.assertEqual(str(~p), "~(x1 < 5)")
        self.assertTrue(p.implies(q) == (~p | q))
        phi = (p & q).finally_(W)
        self.assertTrue(phi.temporal and (phi.lower, phi.upper) == (0, 2))
        self.assertEqual(str(p.until(q, cora.StlInterval(1, 3))), "(x1 < 5 U[1, 3] x2 >= 0)")
        self.assertEqual(str(p.next(1)), "X[1, 1](x1 < 5)")
        self.assertTrue(cora.Stl(True) == cora.Stl(True) and cora.Stl(True) != cora.Stl(False))
        with self.assertRaises(cora.errors.WrongValue):
            p.finally_(cora.StlInterval())
        with self.assertRaises(cora.errors.WrongInputInConstructor):
            p.next(-1)


class Operations(unittest.TestCase):
    def setUp(self):
        x = cora.StlExpr.variables("x", 2)
        self.p, self.q, self.r = x[0] < 5, x[1] >= 0, x[0] > 1

    def test_normal_forms(self):
        # nnf negates the comparison, cnf and dnf distribute, desugar leaves until only
        p, q, r = self.p, self.q, self.r
        ge5 = cora.StlExpr.variables("x", 2)[0] >= 5
        self.assertTrue((~(p & q)).negation_normal_form() == (ge5 | (~q).negation_normal_form()))
        self.assertEqual(str((~p).negation_normal_form()), "x1 >= 5")
        self.assertTrue(((p & q) | r).conjunctive_normal_form() == ((r | p) & (r | q)))
        self.assertTrue(((p | q) & r).disjunctive_normal_form(nnf=True) == ((r & p) | (r & q)))
        self.assertTrue(p.finally_(W).desugar() == cora.Stl(True).until(p, W))
        self.assertTrue((cora.Stl(True) & p).eliminate_true_false() == p)
        self.assertTrue(p.next(1).next(2).combine_next() == p.next(3))

    def test_queries(self):
        # variables, times, clauses and predicates
        p, q, r = self.p, self.q, self.r
        phi = (p & q).finally_(W) & r.next(3)
        self.assertEqual(phi.get_variables(), ["x1", "x2"])
        self.assertEqual(phi.maximum_time(), 3)
        self.assertEqual(phi.minimum_time(), 0)
        res, parts = phi.assign_identifiers()
        self.assertEqual(len(parts), 2)
        windows, points = res.get_times()
        self.assertEqual(windows[0][0], W)
        self.assertEqual(points[1], [3.0])
        self.assertEqual(len(((p & q) & r).get_clauses()), 3)
        self.assertEqual(len((p | q).get_clauses("dnf")), 2)
        with self.assertRaises(cora.errors.WrongValue):
            p.get_clauses("nnf")
        self.assertTrue((p & q).is_predicate() and not (p | q).is_predicate())
        self.assertTrue((p & q).is_linear_predicate())


class Traces(unittest.TestCase):
    def test_trace_checks(self):
        # the values MATLAB's CORA gives on the trace (sin t, cos 3t), t = 0, 0.1, ..., 2
        x1, x2 = cora.StlExpr.variables("x", 2)
        t = np.arange(21) * 0.1
        x = np.stack([np.sin(t), np.cos(3 * t)], axis=1)
        phi = (x1 > 0.8).finally_(cora.StlInterval(0, 1))
        self.assertTrue(phi.model_check_trace(x, t.tolist()))
        self.assertAlmostEqual(phi.robustness(x, t.tolist()), 0.041470984807896683, places=12)
        psi = (x2 > -0.9).globally(cora.StlInterval(0.5, 1.2))
        self.assertFalse(psi.model_check_trace(x, t.tolist()))
        self.assertAlmostEqual(psi.robustness(x, t.tolist()), -0.089992496600445171, places=12)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
