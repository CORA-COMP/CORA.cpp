// test_lean_interval_less_equal - Interval::less_equal against the CORALean oracle (its `le`)
//
// ORACLE: contSet/@interval/le.m
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
    test::oracle::run("interval.le", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n), big = I.enlarge(2.0);
            test::oracle::check(I.less_equal(big), lean::Interval(I).le(lean::Interval(big)));
            test::oracle::check(big.less_equal(I), lean::Interval(big).le(lean::Interval(I)));
            test::oracle::check(I.less_equal(J), lean::Interval(I).le(lean::Interval(J)));
            test::oracle::check(I.less_equal(I), lean::Interval(I).le(lean::Interval(I)));
        }
    });
    return test::finish("lean interval less_equal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
