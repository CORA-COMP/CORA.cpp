// test_lean_interval_plus - Interval::plus against the CORALean oracle
//
// ORACLE: contSet/@interval/plus.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.plus", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(I.plus(J), lean::Interval(I).plus(lean::Interval(J)));
        }
    });
    return test::finish("lean interval plus");
}
