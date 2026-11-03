// test_lean_interval_unite - Interval::unite against the CORALean oracle (its `or`)
//
// ORACLE: contSet/@interval/or.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.or", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(I.unite(J), lean::Interval(I).unite(lean::Interval(J)));
        }
    });
    return test::finish("lean interval unite");
}
