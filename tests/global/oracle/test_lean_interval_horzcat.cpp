// test_lean_interval_horzcat - Interval::horzcat against the CORALean oracle
//
// ORACLE: contSet/@interval/horzcat.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.horzcat", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(Interval::horzcat({I, J}),
                                lean::Interval(I).horzcat(lean::Interval(J)));
        }
    });
    return test::finish("lean interval horzcat");
}
