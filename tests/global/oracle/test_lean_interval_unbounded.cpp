// test_lean_interval_unbounded - Interval::unbounded against the CORALean oracle (its `Inf`)
//
// ORACLE: contSet/@interval/Inf.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.Inf", [](test::oracle::Random &) {
        for (const int n : {1, 2, 20})
            test::oracle::check(Interval::unbounded(n), lean::Interval::unbounded(n));
    });
    return test::finish("lean interval unbounded");
}
