// test_lean_interval_dirac - Interval::dirac against the CORALean oracle
//
// ORACLE: contSet/@interval/dirac.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.dirac", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const int64_t order : {0, 1, 2, 3})
                test::oracle::check(I.dirac(order), lean::Interval(I).dirac(order));
        }
    });
    return test::finish("lean interval dirac");
}
