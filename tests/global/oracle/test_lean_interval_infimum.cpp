// test_lean_interval_infimum - Interval::infimum against the CORALean oracle
//
// ORACLE: contSet/@interval/infimum.m
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
    test::oracle::run("interval.infimum", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.infimum(), lean::Interval(I).infimum());
        }
    });
    return test::finish("lean interval infimum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
