"""test_lean_oracle - three CORA.cpp operations against the CORALean oracle (worked examples)"""
# ORACLE: contSet/@zonotope/supportFunc_.m
# ORACLE: contSet/@zonotope/interval.m
# ORACLE: contSet/@interval/plus.m

import unittest

import cora
import oracle_testing as oracle


@oracle.needs_oracle
class Oracle(unittest.TestCase):
    """Each run is exact on the lean backend at nearest and inside the sound oracle's result."""

    def test_zonotope_support_function(self):
        def body(rnd):
            for n, m in [(2, 3), (5, 30), (20, 200)]:
                Z, d = rnd.zonotope(n, m), rnd.direction(n)
                oracle.check(Z.support_function(d), cora.lean.Zonotope(Z).support_function(d))
        oracle.run("zonotope.supportFunc", body)

    def test_zonotope_interval(self):
        def body(rnd):
            for n, m in [(2, 3), (5, 30), (20, 200)]:
                Z = rnd.zonotope(n, m)
                oracle.check(Z.interval(), cora.lean.Zonotope(Z).interval())
        oracle.run("zonotope.box", body)

    def test_interval_plus(self):
        def body(rnd):
            for n in (2, 5, 20):
                I, J = rnd.box(n), rnd.box(n)
                oracle.check(I.plus(J), cora.lean.Interval(I).plus(cora.lean.Interval(J)))
        oracle.run("interval.plus", body)

    def test_serves(self):
        """The op list of the oracle: a served op, and one that no oracle has."""
        self.assertTrue(cora.lean.serves("zonotope.supportFunc"))
        self.assertFalse(cora.lean.serves("zonotope.noSuchOperation"))


if __name__ == "__main__":
    unittest.main()
