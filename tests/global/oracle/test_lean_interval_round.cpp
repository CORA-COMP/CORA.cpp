// test_lean_interval_round - Interval::round against the CORALean oracle (its `round`)
//
// The oracle rounds to integers (0 decimals), halves away from zero.
//
// ORACLE: contSet/@interval/round.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.round", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const Tensor big = I.sup() * 7.0;
            test::oracle::check(I.round(), lean::Interval(I).round());
            test::oracle::check(Interval(I.inf() * 7.0, big).round(),
                                lean::Interval(Interval(I.inf() * 7.0, big)).round());
        }
        const Interval halves(test::column({-2.5, -0.5, 0.5, 1.5, 2.0}),
                              test::column({-1.5, 0.5, 1.5, 2.5, 3.0}));
        test::oracle::check(halves.round(), lean::Interval(halves).round());
    });
    return test::finish("lean interval round");
}
