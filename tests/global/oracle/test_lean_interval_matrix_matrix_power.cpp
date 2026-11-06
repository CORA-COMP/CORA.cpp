// test_lean_interval_matrix_matrix_power - IntervalMatrix::matrix_power against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/mpower.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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
    test::oracle::run("intervalMatrix.mpower", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 5}) {
            const IntervalMatrix A(rnd.box_matrix(n, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            for (const int k : {0, 1, 2, 3, 5})
                test::oracle::check(A.matrix_power(k), L.matrix_power(k));
        }
    });
    return test::finish("lean interval_matrix matrix_power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
