"""test_nonlinear_property - a VNN-LIB 2.0 property beyond polytopes, decided on boxes of the inputs

The input box from the bounds of single inputs; three-valued evaluation of products, of inputs
compared with outputs and of == / != on boxes (a box with no counterexample may not hold, a point
that holds may); the cells of a variable compared with constants (Y == 0 and Y != 0 never both);
the points in float32; the float32 error bound against a float32 evaluation of a random ReLU
network (never exceeded); the branch and bound unsat, sat (by the check) and unknown (no time).

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""

import unittest


try:
    import numpy as np
    import vnnlib
except ImportError:
    raise unittest.SkipTest("needs numpy and vnnlib") from None

import cora
from cora.nonlinear_property import NonlinearProperty, branch_and_bound, float32_error


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Evaluation(unittest.TestCase):
    def setUp(self):
        self.prop = NonlinearProperty(vnnlib.parse_query_string(TEXT))

    def test_box(self):
        np.testing.assert_array_equal(self.prop.lo, [0, -2])
        np.testing.assert_array_equal(self.prop.hi, [1, 2])

    def test_can_hold(self):
        # x0 * x1 >= Y, Y > 0.5 and (Y != 0 or x1 >= 1)
        p = self.prop
        lo, hi = np.array([[0.0, 1.0], [0.0, -2.0], [0.5, 1.0]]), np.array([[1.0, 2.0], [1.0, 0.0],
                                                                             [1.0, 2.0]])
        ylo, yhi = np.array([[0.4], [0.0], [3.0]]), np.array([[0.6], [1.0], [5.0]])
        # x0 x1 <= 0 below 0.5; x0 x1 <= 2 below 3
        np.testing.assert_array_equal(p.can_hold(lo, hi, ylo, yhi), [True, False, False])

    def test_cells(self):
        # x1 < 1 and Y in [-1, 0.5]: the cell Y = 0 fails x1 >= 1, the others fail Y > 0.5
        p = NonlinearProperty(vnnlib.parse_query_string(CELLS))
        lo, hi = np.array([[0.0, 0.0]]), np.array([[1.0, 0.5]])
        self.assertFalse(p.can_hold(lo, hi, np.array([[-1.0]]), np.array([[0.5]]))[0])
        self.assertTrue(p.can_hold(lo, hi, np.array([[-1.0]]), np.array([[0.6]]))[0])

    def test_holds(self):
        # (1, 1): 1 >= 0.75; (1, 0.25): 0.25 < 0.75
        x = np.array([[1.0, 1.0], [1.0, 0.25]], dtype=np.float32)
        y = np.array([[0.75], [0.75]], dtype=np.float32)
        np.testing.assert_array_equal(self.prop.holds(x, y), [True, False])


class Float32Error(unittest.TestCase):
    def test_bound_holds(self):
        rng = np.random.default_rng(0)
        sizes = [3, 32, 32, 2]
        layers = [(rng.standard_normal((b, a)) * 3, rng.standard_normal(b)) for a, b in
                  zip(sizes[:-1], sizes[1:], strict=True)]
        acts = ["relu", "relu", "identity"]
        # the network in float64 and in float32 on float32 points
        x = rng.uniform(-10, 10, (2000, 3)).astype(np.float32).astype(np.float64)
        exact, single = x, x.astype(np.float32)
        for (W, b), act in zip(layers, acts, strict=True):
            exact = exact @ W.T + b
            single = single @ W.T.astype(np.float32) + b.astype(np.float32)
            if act == "relu":
                exact, single = np.maximum(exact, 0), np.maximum(single, 0)
        bound = float32_error(layers, acts, x, x)
        self.assertTrue(np.all(np.abs(single - exact) <= bound))
        # on boxes around the points, the bound of the points' at least
        box = float32_error(layers, acts, x - 0.01, x + 0.01)
        self.assertTrue(np.all(box >= bound))
        # an activation without a bound: infinite
        self.assertTrue(np.isinf(float32_error(layers, ["tanh"] * 3, x, x)).all())


class BranchAndBound(unittest.TestCase):
    def test_verdicts(self):
        # Y = x0 + x1: x0 x1 <= Y with Y > 0.5 is sat (x = (1, 1)), x0 x1 >= Y + 2 unsat
        layers = [(np.array([[1.0, 1.0]]), np.array([0.0]))]
        net = cora.NeuralNetwork(layers, ["identity"])

        def replay(x):
            return (x @ np.array([[1.0], [1.0]], dtype=np.float32)).astype(np.float32)

        sat = NonlinearProperty(vnnlib.parse_query_string(
            TEXT.replace("(>= (* X[0,0] X[0,1]) Y[0,0])", "(<= (* X[0,0] X[0,1]) Y[0,0])")))
        status, x, _ = branch_and_bound(sat, net, (layers, ["identity"]), 20, replay,
                                        lambda x: bool(sat.holds(x[None], replay(x[None]))[0]))
        self.assertEqual(status, "sat")
        self.assertTrue(sat.holds(x[None], replay(x[None]))[0])
        unsat = NonlinearProperty(vnnlib.parse_query_string(
            TEXT.replace("(>= (* X[0,0] X[0,1]) Y[0,0])", "(>= (* X[0,0] X[0,1]) (+ Y[0,0] 2.0))")))
        status, _, stats = branch_and_bound(unsat, net, (layers, ["identity"]), 20, replay,
                                            lambda x: True)
        self.assertEqual(status, "unsat")
        self.assertGreaterEqual(stats["boxes"], 1)
        self.assertEqual(branch_and_bound(unsat, net, (layers, ["identity"]), 0.0, replay,
                                          lambda x: True)[0], "unknown")


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

HEAD = (
    "(vnnlib-version <2.0>)\n(declare-network f\n (declare-input X real [1, 2])\n"
    " (declare-output Y real [1, 1])\n)\n"
    "(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))\n(assert (and (>= X[0,1] -2.0) (<= X[0,1] 2.0)))\n"
)

TEXT = HEAD + (
    "(assert (>= (* X[0,0] X[0,1]) Y[0,0]))\n(assert (> Y[0,0] 0.5))\n"
    "(assert (or (!= Y[0,0] 0.0) (>= X[0,1] 1.0)))\n"
)

CELLS = HEAD + "(assert (or (and (== Y[0,0] 0.0) (>= X[0,1] 1.0)) (and (!= Y[0,0] 0.0) (> Y[0,0] 0.5))))\n"

# ----------------------------------------  END OF CODE  ----------------------------------------- #
