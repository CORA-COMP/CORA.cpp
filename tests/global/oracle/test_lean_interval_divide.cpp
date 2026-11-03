// test_lean_interval_divide - Interval::divide against the CORALean oracle (its `mrdivide`)
//
// ORACLE: contSet/@interval/mrdivide.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.mrdivide", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const double d : {2.5, -0.75, 1.0})
                test::oracle::check(I.divide(d), lean::Interval(I).mrdivide(d));
        }
    });
    return test::finish("lean interval divide");
}
