// test_lean_interval_matrix_powers - Interval::powers against the CORALean oracle
//
// The power of order initial_order + k is the oracle's powers at offset k from the initial power.
//
// ORACLE: matrixSet/@intervalMatrix/powers.m
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "cora/cont_set/interval/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.powers", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 4}) {
            const Interval A = rnd.box_matrix(n, n), X(rnd.box_matrix(n, n));
            const lean::IntervalMatrix L{lean::Interval(A)};
            for (const int k : {0, 1, 2, 4}) {
                test::oracle::check(mat_exp::powers(A, k + 1).back(),
                                    L.powers(lean::Interval(A), k));
                test::oracle::check(mat_exp::powers(A, k + 3, 3, X).back(),
                                    L.powers(lean::Interval(X), k));
            }
        }
    });
    return test::finish("lean interval_matrix powers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
