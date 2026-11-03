// test_lean_interval_rdivide - Interval::rdivide against the CORALean oracle (its `rdivide`)
//
// The oracle covers a denominator that avoids 0 in every coordinate: here the coordinates
// alternate between a positive and a negative box.
//
// ORACLE: contSet/@interval/rdivide.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

namespace {

/// A box of n dimensions away from 0: [a, a + w] or [-a - w, -a] with a in [0.5, 3].
Interval aux_away_from_zero(test::oracle::Random &rnd, int n) {
    std::vector<double> lo(n), hi(n);
    for (int i = 0; i < n; ++i) {
        const double a = rnd.uniform(0.5, 3), w = rnd.uniform(0, 2);
        lo[i] = i % 2 ? -a - w : a;
        hi[i] = i % 2 ? -a : a + w;
    }
    return {test::column(lo), test::column(hi)};
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.rdivide", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), D = aux_away_from_zero(rnd, n);
            test::oracle::check(I.rdivide(D), lean::Interval(I).rdivide(lean::Interval(D)));
        }
    });
    return test::finish("lean interval rdivide");
}
