// test_lean_interval_matrix_volume - Interval::volume against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/volume.m
// Authors:       Claude Sonnet 5.5, Tobias Ladner
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
    test::oracle::run("intervalMatrix.volume", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] :
             {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}, std::pair{4, 4}}) {
            const Interval A = rnd.box_matrix(m, n);
            const lean::IntervalMatrix L{lean::Interval(A)};
            test::oracle::check(A.volume(), L.volume());
        }
    });
    return test::finish("lean interval_matrix volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
