// test_lean_interval_mink_diff - Interval::mink_diff against the CORALean
//
// oracle (its `minkDiff`), bit for bit at nearest
//
// ORACLE: contSet/@interval/minkDiff.m
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
    test::oracle::run("interval.minkDiff", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // S narrower than I, so the difference is not empty
            const Interval I = rnd.box(n), S = I.enlarge(0.3);
            const Interval R = I.mink_diff(S);
            const lean::Interval O = lean::Interval(I).mink_diff(lean::Interval(S));
            test::oracle::check(R, O);
        }
    });
    return test::finish("lean interval mink_diff");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
