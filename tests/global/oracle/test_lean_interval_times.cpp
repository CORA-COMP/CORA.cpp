// test_lean_interval_times - Interval::times against the CORALean oracle (its `times`)
//
// ORACLE: contSet/@interval/times.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.times", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(I.times(J), lean::Interval(I).times(lean::Interval(J)));
        }
    });
    return test::finish("lean interval times");
}
