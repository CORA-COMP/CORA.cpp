// test_lean_interval_power - Interval::power against the CORALean oracle (its `power`)
//
// The oracle covers the nonnegative integer exponents, which CORA.cpp multiplies in its order.
//
// ORACLE: contSet/@interval/power.m
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
    test::oracle::run("interval.power", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const int exponent : {0, 1, 2, 3, 4, 5, 8, 11})
                test::oracle::check(I.power(exponent), lean::Interval(I).power(exponent));
        }
    });
    return test::finish("lean interval power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
