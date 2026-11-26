"""test_vnnlib - VNN-LIB 1.0 and 2.0 properties as input boxes and unions of output polytopes

The 1.0 reader on boxes, comparisons of two outputs and of an output with a constant (both
directions), a disjunction of conjunctions, and a disjunction of input boxes (ACAS Xu's property
6) merged per box; a 2.0 text, written by to_vnnlib2 from the 1.0 text (exponents as decimals),
reads to the same arrays bit for bit; a 2.0 input fixed by ==, a scaled bound and a box of 2000
inputs; errors for unbounded and empty inputs and for an input constraint that is no box; a 2.0
text with exponents reads too; the specifications as cora objects. Two networks (equal-to): their
inputs and outputs stacked, the relations of inputs, a row of both outputs; products of variables
and comparisons of inputs with outputs are not polytopes.

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   25-November-2026
Reviewed:      ---
"""

import unittest


try:
    import numpy as np
    import vnnlib  # noqa: F401  the reference parser of VNN-LIB 2.0
except ImportError:
    raise unittest.SkipTest("needs numpy and vnnlib") from None

from cora.vnnlib import parse_vnnlib, positional, to_vnnlib2


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Version1(unittest.TestCase):
    def test_box_and_conjunction(self):
        p = parse_vnnlib(SIMPLE, 3, 3)
        self.assertEqual(p.version, 1)
        (case,) = p.cases
        np.testing.assert_array_equal(case.lo, [-0.5, 0.0, 0.25])
        np.testing.assert_array_equal(case.hi, [0.5, 1.0, 0.25])
        ((A, b),) = case.polytopes
        # Y_0 - Y_1 <= 0 and -Y_2 <= -3.5
        np.testing.assert_array_equal(A, [[1, -1, 0], [0, 0, -1]])
        np.testing.assert_array_equal(b, [0, -3.5])

    def test_disjunction(self):
        (case,) = parse_vnnlib(DISJUNCTION, 3, 3).cases
        self.assertEqual(len(case.polytopes), 2)
        np.testing.assert_array_equal(case.polytopes[0][0], [[-1, 0, 1]])
        np.testing.assert_array_equal(case.polytopes[1][0], [[0, 1, 0], [1, -1, 0]])
        np.testing.assert_array_equal(case.polytopes[1][1], [-2, 0])

    def test_disjunction_of_boxes(self):
        # two boxes times two output disjuncts: two cases of two polytopes each
        p = parse_vnnlib(TWO_BOXES, 3, 3)
        self.assertEqual([len(c.polytopes) for c in p.cases], [2, 2])
        np.testing.assert_array_equal([c.lo[0] for c in p.cases], [0, 2])
        index, A, b = p.stacked()
        self.assertEqual((tuple(index), A.shape, b.shape), ((0, 0, 1, 1), (4, 1, 3), (4, 1)))

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "unbounded"):
            parse_vnnlib(DECLARE + "(assert (<= X_0 1))\n(assert (<= Y_0 0))\n", 3, 3)
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_vnnlib(DECLARE + "(assert (<= X_0 0))\n(assert (>= X_0 1))\n", 3, 3)

    def test_specifications(self):
        specs = parse_vnnlib(DISJUNCTION, 3, 3).specifications()
        self.assertEqual([s.type for s in specs], ["unsafe_set", "unsafe_set"])


class Version2(unittest.TestCase):
    def test_conversion_reads_the_same(self):
        # the 2.0 form of each 1.0 text, with the outputs of shape (1, 3)
        for text in (SIMPLE, DISJUNCTION, TWO_BOXES):
            v2 = to_vnnlib2(text, (1, 3), (1, 3))
            self.assertIn("X[0,2]", v2)
            p2 = parse_vnnlib(v2, 3, 3)
            self.assertEqual(p2.version, 2)
            same(self, parse_vnnlib(text, 3, 3), p2)

    def test_exponents_become_decimals(self):
        # 2.0 has no exponent: 1e-06 is written 0.000001, the same double, and the text parses
        text = SIMPLE.replace("(assert", "(assert (<= Y_0 1e-06))\n(assert", 1)
        v2 = to_vnnlib2(text, (1, 3), (1, 3))
        self.assertNotIn("e-06", v2)
        self.assertIn("0.000001", v2)
        same(self, parse_vnnlib(text, 3, 3), parse_vnnlib(v2, 3, 3))

    def test_linear_rows(self):
        # a 2.0 comparison of a linear combination of outputs
        text = (
            "(vnnlib-version <2.0>)\n(declare-network N\n (declare-input X float32 [2])\n"
            " (declare-output Y float32 [2])\n)\n(assert (<= X[0] 1.0))\n(assert (>= X[0] -1.0))\n"
            "(assert (<= X[1] 1.0))\n(assert (>= X[1] -1.0))\n"
            "(assert (>= (+ Y[0] (* 2.0 Y[1])) 1.5))\n"
        )
        ((A, b),) = parse_vnnlib(text, 2, 2).cases[0].polytopes
        np.testing.assert_array_equal(A, [[-1, -2]])
        np.testing.assert_array_equal(b, [-1.5])

    def test_input_atoms(self):
        # an input fixed by ==, a scaled bound and 2000 inputs of a box; a sum of inputs is no box
        n = 2000
        head = (
            "(vnnlib-version <2.0>)\n(declare-network N\n"
            f" (declare-input X float32 [{n}])\n (declare-output Y float32 [1])\n)\n"
        )
        box = "".join(f"(assert (<= X[{i}] 1.0))\n(assert (>= X[{i}] -1.0))\n" for i in range(2, n))
        text = head + "(assert (== X[0] 0.5))\n(assert (<= (* 2.0 X[1]) 1.0))\n"
        text += "(assert (>= X[1] -1.0))\n" + box + "(assert (<= Y[0] 0.0))\n"
        (case,) = parse_vnnlib(text, n, 1).cases
        np.testing.assert_array_equal(case.lo[:3], [0.5, -1.0, -1.0])
        np.testing.assert_array_equal(case.hi[:3], [0.5, 0.5, 1.0])
        # a sum of inputs is a relation of the case (cora.joint_network takes it)
        rel = head + "(assert (<= (+ X[0] X[1]) 1.0))\n(assert (== X[0] 0.5))\n"
        rel += "(assert (<= X[1] 1.0))\n(assert (>= X[1] -1.0))\n" + box + "(assert (<= Y[0] 0.0))\n"
        (case,) = parse_vnnlib(rel, n, 1).cases
        self.assertEqual(case.relations, [({0: 1.0, 1: 1.0}, -1.0)])

    def test_exponents(self):
        # 1.0's exponents (2.5e-05) as 2.0 writes them, positionally; a 2.0 text with them reads too
        text = SIMPLE.replace("0.25", "2.5e-01").replace("3.5", "3.5E+0")
        v2 = to_vnnlib2(text, (1, 3), (1, 3))
        self.assertIn("X[0,2] 0.25)", v2)
        self.assertNotIn("e-01", v2)
        same(self, parse_vnnlib(text, 3, 3), parse_vnnlib(v2, 3, 3))
        raw = v2.replace("0.25)", "2.5e-01)")
        same(self, parse_vnnlib(text, 3, 3), parse_vnnlib(raw, 3, 3))
        self.assertEqual(positional("; 1e-5\n(<= X[0] -1E+3)"), "; 1e-5\n(<= X[0] -1000.0)")


class SeveralNetworks(unittest.TestCase):
    def test_offsets_and_relations(self):
        # f and g (equal-to): inputs and outputs stacked, X_g[0] <= X_f[0] and X_g[1] == X_f[1]
        p = parse_vnnlib(PAIR, None, None)
        self.assertEqual((p.num_inputs, p.num_outputs), (4, 4))
        self.assertEqual([n.name for n in p.networks], ["f", "g"])
        self.assertEqual(p.networks[1].equal_to, "f")
        self.assertEqual(p.networks[1].inputs, [("X_g", (1, 2), 2)])
        self.assertEqual(p.networks[1].outputs, [("Y_g", (1, 2), 2)])
        (case,) = p.cases
        np.testing.assert_array_equal(case.lo, [0, -1, -np.inf, -np.inf])
        np.testing.assert_array_equal(case.hi, [1, 1, np.inf, np.inf])
        rows = sorted((tuple(sorted(r.items())), c + 0.0) for r, c in case.relations)
        self.assertEqual(rows, [(((0, -1.0), (2, 1.0)), 0.0), (((1, -1.0), (3, 1.0)), 0.0),
                                (((1, 1.0), (3, -1.0)), 0.0)])
        # Y_f[0,1] < Y_g[0,1]: the row of the stacked outputs
        ((A, b),) = case.polytopes
        np.testing.assert_array_equal(A, [[0, 1, 0, -1]])
        np.testing.assert_array_equal(b + 0.0, [0])

    def test_nonlinear_is_not_implemented(self):
        # a product of variables or an input compared with an output: no polytope
        text = PAIR.split("(assert (< Y_f")[0] + "(assert (<= (* Y_f[0,0] Y_g[0,0]) 1.0))\n"
        with self.assertRaisesRegex(NotImplementedError, "nonlinear"):
            parse_vnnlib(text, None, None)
        text = PAIR.split("(assert (< Y_f")[0] + "(assert (<= Y_f[0,0] X_f[0,0]))\n"
        with self.assertRaisesRegex(NotImplementedError, "mixes"):
            parse_vnnlib(text, None, None)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

# f and g (equal-to): a box of X_f, X_g[0] <= X_f[0], X_g[1] == X_f[1], Y_f[0,1] < Y_g[0,1]
PAIR = (
    "(vnnlib-version <2.0>)\n"
    "(declare-network f\n (declare-input X_f real [1, 2])\n (declare-output Y_f real [1, 2])\n)\n"
    "(declare-network g\n (equal-to f)\n (declare-input X_g real [1, 2])\n"
    " (declare-output Y_g real [1, 2])\n)\n"
    "(assert (and (<= X_f[0,0] 1.0) (>= X_f[0,0] 0.0)))\n"
    "(assert (and (<= X_f[0,1] 1.0) (>= X_f[0,1] -1.0)))\n"
    "(assert (>= X_f[0,0] X_g[0,0]))\n(assert (== X_f[0,1] X_g[0,1]))\n"
    "(assert (< Y_f[0,1] Y_g[0,1]))\n"
)

# the 1.0 declarations of 3 inputs and 3 outputs
DECLARE = "".join(f"(declare-const X_{i} Real)\n" for i in range(3)) + "".join(
    f"(declare-const Y_{j} Real)\n" for j in range(3)
)

BOX = (
    "(assert (<= X_0 0.5))\n(assert (>= X_0 -0.5))\n(assert (<= X_1 1))\n(assert (>= X_1 0))\n"
    "(assert (<= X_2 0.25))\n(assert (>= X_2 0.25))\n"
)

SIMPLE = DECLARE + "; a comment\n" + BOX + "(assert (<= Y_0 Y_1))\n(assert (>= Y_2 3.5))\n"

DISJUNCTION = DECLARE + BOX + "(assert (or\n  (and (>= Y_0 Y_2))\n  (and (<= Y_1 -2) (>= Y_1 Y_0))\n))\n"

TWO_BOXES = (
    DECLARE + "(assert (or (and (<= X_0 1) (>= X_0 0) (<= X_1 1) (>= X_1 0)) "
    "(and (<= X_0 3) (>= X_0 2) (<= X_1 1) (>= X_1 0))))\n(assert (<= X_2 0))\n(assert (>= X_2 0))\n"
    "(assert (or (and (<= Y_1 Y_0)) (and (<= Y_2 Y_0))))\n"
)


def same(test, p, q):
    """Two properties with the same cases, bit for bit."""
    test.assertEqual(len(p.cases), len(q.cases))
    for a, b in zip(p.cases, q.cases, strict=True):
        test.assertTrue(np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi))
        test.assertEqual(len(a.polytopes), len(b.polytopes))
        for (A1, b1), (A2, b2) in zip(a.polytopes, b.polytopes, strict=True):
            test.assertTrue(np.array_equal(A1, A2) and np.array_equal(b1 + 0.0, b2 + 0.0))

# ----------------------------------------  END OF CODE  ----------------------------------------- #
