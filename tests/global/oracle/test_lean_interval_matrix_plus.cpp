// test_lean_interval_matrix_plus - IntervalMatrix::plus against the CORALean oracle
//
// The sum with an interval matrix is the oracle's plus; a matrix or a number is the same with a
// point matrix.
//
// ORACLE: matrixSet/@intervalMatrix/plus.m
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.plus", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] : {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n)), B(rnd.box_matrix(m, n));
            const lean::IntervalMatrix LA{lean::Interval(A.interval())};
            test::oracle::check(A.plus(B), LA.plus(lean::Interval(B.interval())));
            const Tensor M = rnd.matrix(m, n);
            test::oracle::check(A.plus(M), LA.plus(lean::Interval::from(M, M)));
            const Tensor S = M.full_like(M.shape(), 0.75);
            test::oracle::check(A.plus(0.75), LA.plus(lean::Interval::from(S, S)));
        }
    });
    return test::finish("lean interval_matrix plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
