// test_lean_interval_min - Interval::min against the CORALean oracle (its `min`)
//
// ORACLE: contSet/@interval/min.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.min", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(I.min(J), lean::Interval(I).min(lean::Interval(J)));
        }
    });
    return test::finish("lean interval min");
}
