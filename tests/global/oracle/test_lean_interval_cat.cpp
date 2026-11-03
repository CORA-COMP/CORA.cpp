// test_lean_interval_cat - Interval::cat against the CORALean oracle (its `cat`)
//
// ORACLE: contSet/@interval/cat.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.cat", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            for (const int dim : {1, 2})
                test::oracle::check(Interval::cat(dim, {I, J}),
                                    lean::Interval(I).cat(lean::Interval(J), dim));
        }
    });
    return test::finish("lean interval cat");
}
