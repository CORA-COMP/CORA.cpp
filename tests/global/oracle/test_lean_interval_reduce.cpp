// test_lean_interval_reduce - Interval::reduce against the CORALean oracle (its `reduce`)
//
// ORACLE: contSet/@interval/reduce.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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
    test::oracle::run("interval.reduce", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const int order : {1, 3})
                test::oracle::check(I.reduce(order), lean::Interval(I).reduce(order));
        }
    });
    return test::finish("lean interval reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
