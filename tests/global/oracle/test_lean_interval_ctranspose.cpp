// test_lean_interval_ctranspose - Interval::ctranspose against the CORALean oracle
//
// ORACLE: contSet/@interval/ctranspose.m
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
    test::oracle::run("interval.ctranspose", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), M = Interval::horzcat({I, rnd.box(n), rnd.box(n)});
            test::oracle::check(I.ctranspose(), lean::Interval(I).ctranspose());
            test::oracle::check(M.ctranspose(), lean::Interval(M).ctranspose());
        }
    });
    return test::finish("lean interval ctranspose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
