// test_lean_interval_enlarge - Interval::enlarge against the CORALean oracle
//
// ORACLE: contSet/@interval/enlarge.m
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
    test::oracle::run("interval.enlarge", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const double factor : {0.0, 0.5, 1.0, 2.5})
                test::oracle::check(I.enlarge(factor), lean::Interval(I).enlarge(factor));
        }
    });
    return test::finish("lean interval enlarge");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
