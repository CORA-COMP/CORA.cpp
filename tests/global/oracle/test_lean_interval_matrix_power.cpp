// test_lean_interval_matrix_power - Interval::matrix_power against the CORALean oracle (its
//
// `mpower`)
//
// ORACLE: contSet/@interval/mpower.m
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.mpower", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 5}) {
            const Tensor A = rnd.matrix(n, n);
            const Interval I(A - 0.1, A + 0.2);
            for (const int k : {0, 1, 2, 3, 5})
                test::oracle::check(I.matrix_power(k), lean::Interval(I).mpower(k));
        }
    });
    return test::finish("lean interval matrix_power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
