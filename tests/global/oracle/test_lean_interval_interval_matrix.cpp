// test_lean_interval_interval_matrix - Interval::interval_matrix against the CORALean oracle
//
// ORACLE: contSet/@interval/intervalMatrix.m
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
    test::oracle::run("interval.intervalMatrix", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // the oracle's constructor at CORA.cpp's center and radius (exact doubles)
            const Interval I = rnd.box(n);
            const Interval M = Interval::horzcat({I, rnd.box(n), rnd.box(n)});
            for (const Interval &B : {I, M})
                test::oracle::check(B.interval_matrix(),
                                    lean::Interval::interval_matrix(lean::Tensor::from(B.center()),
                                                                    lean::Tensor::from(B.rad())));
        }
    });
    return test::finish("lean interval interval_matrix");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
