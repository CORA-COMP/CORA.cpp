// test_lean_interval_isequal - Interval::isequal against the CORALean oracle (its `isequal`)
//
// ORACLE: contSet/@interval/isequal.m
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
    test::oracle::run("interval.isequal", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(I.isequal(I, 0), lean::Interval(I).isequal(lean::Interval(I), 0));
            test::oracle::check(I.isequal(J, 0), lean::Interval(I).isequal(lean::Interval(J), 0));
            const Interval K(I.inf(), I.sup() + 1e-9);
            test::oracle::check(I.isequal(K, 0), lean::Interval(I).isequal(lean::Interval(K), 0));
        }
    });
    return test::finish("lean interval isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
