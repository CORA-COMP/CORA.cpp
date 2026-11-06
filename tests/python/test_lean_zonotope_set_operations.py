"""test_lean_zonotope_set_operations - Zonotope operations against the CORALean oracle: abs, box,

cart_prod, compact, constraints_satisfied, enclose, convex_hull, contains, dominant_directions.

Authors:       Tobias Ladner
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
# ORACLE: contSet/@zonotope/abs.m
# ORACLE: contSet/@zonotope/box.m
# ORACLE: contSet/@zonotope/cartProd_.m
# ORACLE: contSet/@zonotope/compact_.m
# ORACLE: contSet/@zonotope/constrSat.m
# ORACLE: contSet/@zonotope/contains_.m
# ORACLE: contSet/@zonotope/convHull_.m
# ORACLE: contSet/@zonotope/dominantDirections.m
# ORACLE: contSet/@zonotope/enclose.m

import unittest

import cora
import numpy as np
import oracle_testing as oracle

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

SIZES = [(2, 3), (5, 30), (20, 200)]


def sorted_columns(matrix):
    """The columns of a matrix as a sorted list, so that another order of them compares equal."""
    return sorted(tuple(column) for column in np.asarray(matrix).T)


def check_set(mine, theirs):
    """Exact: the same bits; sound: inside the oracle's enclosure, by the interval hulls (the
    oracle's certified containment of a zonotope in a zonotope is slow for large ones)."""
    if oracle.mode() == "exact":
        oracle.check(mine, theirs)
    else:
        oracle.check(mine.interval(), theirs.enclosure().interval())


def check_hull(mine, theirs):
    """The box of CORA.cpp's cubic map against the oracle's, to a relative 1e-10: CORA lists
    N(N+1)(N+2)/6 generators, CORALean N^3, and they sum in another order."""
    theirs_box, box = theirs.interval(), mine.interval()
    np.testing.assert_allclose(box.inf, theirs_box.inf.gather(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(box.sup, theirs_box.sup.gather(), rtol=1e-10, atol=1e-10)


def check_same_generators(mine, theirs):
    """CORA lists (G+G2)/2, (c-c2)/2, (G-G2)/2, CORALean's linComb (G+G2)/2, (G-G2)/2, (c-c2)/2: the
    same center and generators in another order."""
    if oracle.mode() != "exact":
        return check_set(mine, theirs)
    other = theirs.gather()[0]
    np.testing.assert_array_equal(mine.center, other.center)
    assert sorted_columns(mine.generators) == sorted_columns(other.generators)


# ============================================  MAIN  ============================================ #

@oracle.needs_oracle
class Oracle(unittest.TestCase):
    def test_abs(self):
        def body(rnd):
            for n, m in SIZES:
                Z = rnd.zonotope(n, m)
                check_set(Z.abs(), cora.lean.Zonotope(Z).abs())
        oracle.run("zonotope.abs", body)

    def test_box(self):
        def body(rnd):
            for n, m in SIZES:
                Z = rnd.zonotope(n, m)
                oracle.check(Z.box().interval(), cora.lean.Zonotope(Z).interval())
        oracle.run("zonotope.box", body)

    def test_cart_prod(self):
        def body(rnd):
            for n, m in SIZES:
                Z, Z2 = rnd.zonotope(n, m), rnd.zonotope(n + 1, m + 2)
                check_set(Z.cart_prod(Z2), cora.lean.Zonotope(Z).cart_prod(cora.lean.Zonotope(Z2)))
        oracle.run("zonotope.cartProd", body)

    def test_compact(self):
        def body(rnd):
            for n, m in SIZES:
                Z = rnd.zonotope(n, m)
                G = Z.generators
                zero = np.zeros((n, 1))
                Zz = cora.Zonotope(Z.center, np.hstack([G[:, :2], zero, G[:, 2:], zero]))
                check_set(Zz.compact("zeros", 0.0), cora.lean.Zonotope(Zz).compact("zeros", 0.0))
        oracle.run("zonotope.compact", body)

    def test_constraints_satisfied(self):
        def body(rnd):
            for n, m in SIZES:
                Z = rnd.zonotope(n, m)
                C = rnd.matrix(3, n)
                sup = (C @ Z).interval().sup
                for margin in (1e-3, -1e-3):
                    oracle.check(Z.constraints_satisfied(C, sup + margin),
                                 cora.lean.Zonotope(Z).constr_sat(C, sup + margin))
        oracle.run("zonotope.constrSat", body)

    def test_enclose_and_convex_hull(self):
        def body(rnd):
            for n, m in SIZES:
                Z, Z2 = rnd.zonotope(n, m), rnd.zonotope(n, m)
                check_same_generators(Z.enclose(Z2), cora.lean.Zonotope(Z).enclose(cora.lean.Zonotope(Z2)))
                check_same_generators(Z.convex_hull(Z2),
                                      cora.lean.Zonotope(Z).convex_hull(cora.lean.Zonotope(Z2)))
        oracle.run("zonotope.enclose", body)

    def test_contains(self):
        def body(rnd):
            for n, m in [(2, 3), (3, 5)]:
                Z = rnd.zonotope(n, m)
                # half of Z about its center is in Z, one and a half times is not
                inner = cora.Zonotope(Z.center, 0.5 * Z.generators)
                outer = cora.Zonotope(Z.center, 1.5 * Z.generators)
                for S in (inner, outer):
                    theirs = cora.lean.Zonotope(Z).contains(cora.lean.Zonotope(S))
                    oracle.check(Z.contains(S, "approx:st"), theirs)
        oracle.run("zonotope.contains", body)

    def test_cubic_map(self):
        def body(rnd):
            for n, m in [(2, 2), (3, 3), (3, 6)]:
                Z = rnd.zonotope(n, m)
                T = [[rnd.matrix(n, n) for _ in range(n)] for _ in range(2)]
                check_hull(Z.cubic_map(T), cora.lean.Zonotope(Z).cub_map(T))
        oracle.run("zonotope.cubMap", body)

    def test_dominant_directions(self):
        def body(rnd):
            if oracle.mode() != "exact":
                return  # a choice of columns has no enclosure
            for n, m in [(2, 3), (3, 9), (4, 12)]:
                Z = rnd.zonotope(n, m)
                theirs = cora.lean.Zonotope(Z).dominant_directions()
                self.assertIsNotNone(theirs)
                self.assertEqual(sorted_columns(Z.dominant_directions()), sorted_columns(theirs.gather()))
        oracle.run("zonotope.dominantDirections", body)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
