// test_lean_interval_matrix_infinity_norm - IntervalMatrix::infinity_norm against the CORALean
//
// oracle
//
// ORACLE: matrixSet/@intervalMatrix/infNorm.m
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
    test::oracle::run("intervalMatrix.infNorm", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] :
             {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}, std::pair{4, 4}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            test::oracle::check(A.infinity_norm(), L.infinity_norm());
        }
    });
    return test::finish("lean interval_matrix infinity_norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
