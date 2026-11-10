"""test_lean_oracle - the wire of cora.lean: one round trip through the oracle per wrapper kind

The numerics are the C++ oracle tests' (tests/global/oracle/); here the set wrappers (Interval,
Zonotope, IntervalMatrix, EmptySet, Fullspace) cross to the oracle and back once, on the lean
backend at dtype nearest, where the answer is CORA.cpp's bit for bit (test_lean.py: Tensor,
Certified and LinearSys at binary64).

Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
Written:       02-November-2026
Last update:   07-November-2026
Reviewed:      ---
"""
import contextlib
import os
import unittest
import warnings

import cora
import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

# Skips without CORACPP_ORACLE; with CORACPP_REQUIRE_ORACLE (CI) the tests run and fail instead.
needs_oracle = unittest.skipUnless(
    os.environ.get("CORACPP_ORACLE") or os.environ.get("CORACPP_REQUIRE_ORACLE", "0") not in ("", "0"),
    "needs CORACPP_ORACLE")


def same_bits(test, a, b):
    """a and b as float64 arrays of one shape and the same bits (-0 differs from 0)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    test.assertEqual(a.shape, b.shape)
    test.assertTrue(np.array_equal(a.view(np.uint64), b.view(np.uint64)), f"{a} != {b}")


def column(x):
    """A result of the oracle as a vector (its matrices are (n, 1))."""
    return np.asarray(x, dtype=np.float64).reshape(-1)


# ============================================  MAIN  ============================================ #

@needs_oracle


class RoundTrip(unittest.TestCase):
    """CORA.cpp on the lean backend and the oracle at nearest give the same bits."""

    def setUp(self):
        scopes = contextlib.ExitStack()
        scopes.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore", cora.CORAWarning)  # nearest has no guarantee, by design
        scopes.enter_context(cora.backend("lean"))
        scopes.enter_context(cora.lean.dtype("nearest"))
        self.addCleanup(scopes.close)
        self.rng = np.random.default_rng(7)

    def bounds(self, *shape):
        """Random lower and upper bounds of one shape."""
        c, r = self.rng.standard_normal(shape), self.rng.uniform(0, 1, shape)
        return c - r, c + r

    def box(self, n):
        return cora.Interval(*self.bounds(n))

    def test_zonotope(self):
        Z = cora.Zonotope(self.rng.standard_normal(3), self.rng.standard_normal((3, 5)))
        d = self.rng.standard_normal(3)
        same_bits(self, Z.support_function(d), cora.lean.Zonotope(Z).support_function(d))

    def test_interval(self):
        I, J = self.box(4), self.box(4)
        theirs = cora.lean.Interval(I).plus(cora.lean.Interval(J)).gather()
        same_bits(self, (I + J).inf, column(theirs.inf))
        same_bits(self, (I + J).sup, column(theirs.sup))

    # an interval matrix: its lower bounds
    def test_interval_matrix(self):
        A = cora.IntervalMatrix(cora.Interval.matrix(*self.bounds(2, 3)))
        L = cora.lean.IntervalMatrix(cora.lean.Interval(A.interval))
        same_bits(self, A.infimum, np.asarray(L.infimum().gather(), dtype=np.float64))

    def test_empty_set_and_fullspace(self):
        self.assertEqual(cora.lean.EmptySet.empty(3).dim, cora.EmptySet(3).dim)
        self.assertEqual(cora.lean.Fullspace(3).is_bounded(), cora.Fullspace(3).is_bounded())

    def test_serves(self):
        """The op list of the oracle: a served op, and one that no oracle has."""
        self.assertTrue(cora.lean.serves("zonotope.supportFunc"))
        self.assertFalse(cora.lean.serves("zonotope.noSuchOperation"))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
