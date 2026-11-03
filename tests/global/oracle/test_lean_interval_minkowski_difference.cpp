// test_lean_interval_minkowski_difference - Interval::minkowski_difference against the CORALean
// oracle (its `minkDiff`)
//
// The oracle's difference is an inner approximation (the lower bound rounded up, the upper one
// down), so in the sound mode the oracle's box lies in CORA.cpp's, not the other way round.
//
// ORACLE: contSet/@interval/minkDiff.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.minkDiff", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // S narrower than I, so the difference is not empty
            const Interval I = rnd.box(n), S = I.enlarge(0.3);
            const Interval R = I.minkowski_difference(S);
            const lean::Interval O = lean::Interval(I).mink_diff(lean::Interval(S));
            if (test::oracle::mode() == test::oracle::Mode::exact) {
                test::oracle::check_exact("interval.minkDiff", R, O);
            } else {
                const auto [inf, sup] = O.gather();
                test::oracle::check_enclosed("interval.minkDiff", Interval(inf, sup), lean::Interval(R));
            }
        }
    });
    return test::finish("lean interval minkowski_difference");
}
