// test_lean_interval_origin - Interval::origin against the CORALean oracle (its `origin`)
//
// ORACLE: contSet/@interval/origin.m
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
    test::oracle::run("interval.origin", [](test::oracle::Random &) {
        for (const int n : {1, 2, 5, 20})
            test::oracle::check(Interval::origin(n), lean::Interval::origin(n));
    });
    return test::finish("lean interval origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
