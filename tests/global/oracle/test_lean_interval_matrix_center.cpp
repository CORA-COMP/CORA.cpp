// test_lean_interval_matrix_center - Interval::center against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/center.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.center", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] :
             {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}, std::pair{4, 4}}) {
            const Interval A = rnd.box_matrix(m, n);
            const lean::IntervalMatrix L{lean::Interval(A)};
            test::oracle::check(A.center(), L.center());
        }
    });
    return test::finish("lean interval_matrix center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
