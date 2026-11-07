// test_lean_interval_norm - Interval::norm against the CORALean oracle (its `norm`)
//
// The oracle answers the squared length rounded up (its `normSq`, the float definition has no
// square root). On the lean backend CORA.cpp's norm is the correctly rounded root of the same sum
// of squares, so it equals the root of the oracle's number bit for bit.
//
// ORACLE: contSet/@interval/norm_.m
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.norm", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const double root = std::sqrt(lean::Interval(I).norm());
            test::oracle::check("interval.norm", I.norm().data()[0], root);
        }
    });
    return test::finish("lean interval norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
