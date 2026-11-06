// test_lean_interval_is_bounded - Interval::is_bounded against the CORALean oracle
//
// ORACLE: contSet/@interval/isBounded.m
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
    test::oracle::run("interval.isBounded", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.is_bounded(), lean::Interval(I).is_bounded());
        }
    });
    return test::finish("lean interval is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
