// test_lean_interval_diag - Interval::diag against the CORALean oracle
//
// ORACLE: contSet/@interval/diag.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.diag", [](test::oracle::Random &rnd) {
        for (const int n : {1, 2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.diag(), lean::Interval(I).diag());
        }
    });
    return test::finish("lean interval diag");
}
