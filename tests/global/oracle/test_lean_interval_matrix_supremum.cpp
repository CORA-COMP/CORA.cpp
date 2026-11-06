// test_lean_interval_matrix_supremum - IntervalMatrix::supremum against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/supremum.m
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
    test::oracle::run("intervalMatrix.supremum", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] :
             {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}, std::pair{4, 4}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            test::oracle::check(A.supremum(), L.supremum());
        }
    });
    return test::finish("lean interval_matrix supremum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
