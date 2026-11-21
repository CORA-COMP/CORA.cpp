// test_lean_interval_sign_support_function - Interval::sign and support_function against the
//
// CORALean oracle, in CORA's own forms
//
// The oracle has ops of its own for sign ([sign inf, sign sup], `interval.signCora`) and for the
// support function (d'c + |d|'r, the center rounded both ways, `interval.supportFuncCora`); at
// dtype nearest they are CORA's, bit for bit. The oracle's plain `interval.sign` and
// `interval.supportFunc` are other definitions (the first is 0 below a positive box, the second a
// maximum per coordinate), and are not compared here.
//
// ORACLE: contSet/@interval/sign.m
// ORACLE: contSet/@interval/supportFunc_.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A box of n dimensions: unit-scale bounds, with 0 as a bound in the first and last coordinate
/// when `zeros`, which are the cases where a sign differs most.
Interval aux_box(test::oracle::Random &rnd, int n, bool zeros) {
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rnd.scalar();
        b[i] = a[i] + rnd.uniform(0, 1);
    }
    if (zeros) {
        b[0] = std::max(b[0], 0.0);
        a[0] = 0.0;
        b[n - 1] = 0.0;
        a[n - 1] = std::min(a[n - 1], -0.5);
    }
    return {test::vec(a), test::vec(b)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.signCora", [](test::oracle::Random &rnd) {
        for (const bool zeros : {false, true})
            for (const int n : {2, 4, 6}) {
                const Interval I = aux_box(rnd, n, zeros);
                test::oracle::check(I.sign(), lean::Interval(I).sign_cora());
            }
    });
    test::oracle::run("interval.supportFuncCora", [](test::oracle::Random &rnd) {
        for (const bool zeros : {false, true})
            for (const int n : {2, 4, 6}) {
                const Interval I = aux_box(rnd, n, zeros);
                const Tensor d = rnd.direction(n);
                test::oracle::check(I.support_function(d),
                                    lean::Interval(I).support_function_cora(d));
            }
    });
    return test::finish("lean interval sign and support function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
