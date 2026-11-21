// test_lean_interval_sqrt - Interval::sqrt against the CORALean oracle, bit for bit
//
// The oracle's `interval.sqrtCora` is CORA's sqrt of a nonnegative box (the machine root at both
// ends); the oracle's plain `interval.sqrt` rounds outward and is not compared. Boxes of
// nonnegative numbers only, with bounds of several scales.
//
// ORACLE: contSet/@interval/sqrt.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <vector>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("interval.sqrtCora", [](oracle::Random &rnd) {
        for (const double scale : {1e-3, 1.0, 1e3})
            for (const int n : {1, 4, 6}) {
                std::vector<double> a(n), b(n);
                for (int i = 0; i < n; ++i) {
                    const double u = rnd.uniform(0, 10);
                    a[i] = u * u * scale;
                    b[i] = a[i] + rnd.uniform(0, 5);
                }
                const Interval I(test::vec(a), test::vec(b));
                oracle::check(I.sqrt(), lean::Interval(I).sqrt_cora());
            }
    });
    return test::finish("lean interval sqrt");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
