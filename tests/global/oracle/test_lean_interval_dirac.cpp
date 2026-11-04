// test_lean_interval_dirac - Interval::dirac against the CORALean oracle
//
// A box [2^-100, 1] misses 0: its dirac is exactly 0. The oracles before batch 5 compare with a
// margin that rounds away at nearest and answer the other way, so with them the exact mode leaves
// it out.
//
// ORACLE: contSet/@interval/dirac.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.dirac", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const int64_t order : {0, 1, 2, 3})
                test::oracle::check(I.dirac(order), lean::Interval(I).dirac(order));
        }
        if (test::oracle::mode() == test::oracle::Mode::sound ||
            test::oracle::nearest_is_strict()) {
            const Interval tiny(test::column({std::ldexp(1.0, -100)}), test::column({1.0}));
            test::oracle::check(tiny.dirac(0), lean::Interval(tiny).dirac(0));
        }
    });
    return test::finish("lean interval dirac");
}
