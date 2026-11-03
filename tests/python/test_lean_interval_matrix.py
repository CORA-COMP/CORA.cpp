"""test_lean_interval_matrix - IntervalMatrix against the CORALean oracle (cora.lean.IntervalMatrix)

norm is not here as ORACLE: the Frobenius norm is a square root that the oracle bounds from above by
Newton steps, so CORA.cpp's correctly rounded root is inside the bound but not its bits."""
# ORACLE: matrixSet/@intervalMatrix/abs.m
# ORACLE: matrixSet/@intervalMatrix/center.m
# ORACLE: matrixSet/@intervalMatrix/contains.m
# ORACLE: matrixSet/@intervalMatrix/dependentTerms.m
# ORACLE: matrixSet/@intervalMatrix/exactSquare.m
# ORACLE: matrixSet/@intervalMatrix/infNorm.m
# ORACLE: matrixSet/@intervalMatrix/infimum.m
# ORACLE: matrixSet/@intervalMatrix/intervalMatrix.m
# ORACLE: matrixSet/@intervalMatrix/mpower.m
# ORACLE: matrixSet/@intervalMatrix/mtimes.m
# ORACLE: matrixSet/@intervalMatrix/plus.m
# ORACLE: matrixSet/@intervalMatrix/powers.m
# ORACLE: matrixSet/@intervalMatrix/rad.m
# ORACLE: matrixSet/@intervalMatrix/representsa.m
# ORACLE: matrixSet/@intervalMatrix/supremum.m
# ORACLE: matrixSet/@intervalMatrix/vertices.m
# ORACLE: matrixSet/@intervalMatrix/volume.m

import unittest

import cora
import numpy as np
import oracle_testing as oracle

SHAPES = ((2, 2), (3, 4), (5, 1), (4, 4))


def box_matrix(rnd, rows, cols):
    """An interval matrix: a center and a radius in [0, 1] per entry."""
    c, r = rnd.matrix(rows, cols), rnd.gen.uniform(0, 1, (rows, cols))
    return cora.IntervalMatrix(cora.Interval.matrix(c - r, c + r))


def lean_of(A):
    return cora.lean.IntervalMatrix(cora.lean.Interval(A.interval))


def point(M):
    return cora.Interval.matrix(M, M)


@oracle.needs_oracle
class Oracle(unittest.TestCase):
    def each(self, op, body, shapes=SHAPES):
        def run(rnd):
            for rows, cols in shapes:
                A = box_matrix(rnd, rows, cols)
                body(rnd, A, lean_of(A))
        oracle.run(op, run)

    def test_the_matrices_of_the_set(self):
        self.each("intervalMatrix.abs", lambda rnd, A, L: oracle.check(A.abs(), L.abs()))
        self.each("intervalMatrix.infimum", lambda rnd, A, L: oracle.check(A.infimum, L.infimum()))
        self.each("intervalMatrix.center",
                  lambda rnd, A, L: oracle.check(point(A.center), L.center()))

    def test_contains(self):
        def body(rnd, A, L):
            for M in (A.center, A.infimum, A.interval.sup, A.center + 5.0):
                for tol in (0.0, 0.0009765625):
                    oracle.check(A.contains(M, tol=tol), L.contains(M, tol))
        self.each("intervalMatrix.contains", body)

    def test_squares_and_powers(self):
        squares = ((2, 2), (3, 3), (5, 5))
        self.each("intervalMatrix.exactSquare",
                  lambda rnd, A, L: oracle.check(A.exact_square().interval, L.exact_square()), squares)

        def terms(rnd, A, L):
            for r in (1.0, 0.5, 0.125):
                oracle.check(tuple(x.interval for x in A.dependent_terms(r)), L.dependent_terms(r))
        self.each("intervalMatrix.dependentTerms", terms, squares)

        def powers(rnd, A, L):
            for k in (0, 1, 2, 3, 5):
                oracle.check(A.matrix_power(k).interval, L.matrix_power(k))
        self.each("intervalMatrix.mpower", powers, squares)

    def test_products(self):
        def run(rnd):
            for m, n, p in ((2, 2, 2), (3, 4, 2), (1, 5, 3)):
                A, B = box_matrix(rnd, m, n), box_matrix(rnd, n, p)
                oracle.check((A @ B).interval, lean_of(A).matmul(cora.lean.Interval(B.interval)))
                M = rnd.matrix(n, p)
                oracle.check((A @ M).interval, lean_of(A).matmul(cora.lean.Interval(point(M))))
        oracle.run("intervalMatrix.mtimes", run)

    def test_sums_and_queries(self):
        self.each("intervalMatrix.rad", lambda rnd, A, L: oracle.check(A.rad, L.rad()))
        self.each("intervalMatrix.supremum", lambda rnd, A, L: oracle.check(A.supremum, L.supremum()))
        self.each("intervalMatrix.volume", lambda rnd, A, L: oracle.check(A.volume(), L.volume()))

        def plus(rnd, A, L):
            B = box_matrix(rnd, A.rows, A.cols)
            M = rnd.matrix(A.rows, A.cols)
            oracle.check((A + B).interval, L.plus(cora.lean.Interval(B.interval)))
            oracle.check((A + M).interval, L.plus(cora.lean.Interval(point(M))))
        self.each("intervalMatrix.plus", plus)

        def represents(rnd, A, L):
            M = rnd.matrix(A.rows, A.cols)
            # at dtype nearest the oracle's emptySet is true for inf = sup: only boxes of positive radius
            for S in (A, cora.IntervalMatrix(M), cora.IntervalMatrix(0.0 * M)):
                for kind in ("emptySet", "point", "origin"):
                    if kind != "emptySet" or S is A:
                        oracle.check(S.representsa(kind, tol=0.0), lean_of(S).representsa(kind, 0.0))
        self.each("intervalMatrix.representsa", represents)

    def test_powers(self):
        def powers(rnd, A, L):
            X = box_matrix(rnd, A.rows, A.cols)
            for k in (0, 1, 2, 4):
                oracle.check(A.powers(k + 1)[-1].interval, L.powers(cora.lean.Interval(A.interval), k))
                oracle.check(A.powers(k + 3, 3, X)[-1].interval, L.powers(cora.lean.Interval(X.interval), k))
        self.each("intervalMatrix.powers", powers, ((2, 2), (3, 3), (4, 4)))

    def test_vertices(self):
        def vertices(rnd, A, L):
            theirs = [np.asarray(V.gather()) for V in L.vertices().V]
            theirs.sort(key=lambda M: tuple(M.T.reshape(-1)))
            mine = A.vertices()[0]
            np.testing.assert_array_equal(mine, np.stack(theirs))
        self.each("intervalMatrix.vertices", vertices, ((2, 2), (1, 3), (3, 2)))

    def test_the_infinity_norm(self):
        self.each("intervalMatrix.infNorm",
                  lambda rnd, A, L: oracle.check(A.infinity_norm(), L.infinity_norm()))

    def test_the_constructor(self):
        def run(rnd):
            for rows, cols in SHAPES:
                c, r = rnd.matrix(rows, cols), rnd.gen.uniform(0, 1, (rows, cols))
                oracle.check(cora.IntervalMatrix(c, r).interval,
                             cora.lean.Interval.interval_matrix(c, r))
        oracle.run("interval.intervalMatrix", run)


if __name__ == "__main__":
    unittest.main()
