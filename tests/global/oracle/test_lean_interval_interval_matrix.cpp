// test_lean_interval_interval_matrix - Interval::interval_matrix against the CORALean oracle
//
// ORACLE: contSet/@interval/intervalMatrix.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.intervalMatrix", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // the oracle's constructor at the center and radius of CORA.cpp (exact doubles); the
            // center and the radius have their own oracle tests
            const Interval I = rnd.box(n);
            const Interval M = Interval::horzcat({I, rnd.box(n), rnd.box(n)});
            for (const Interval &B : {I, M})
                test::oracle::check(B.interval_matrix(),
                                    lean::Interval::interval_matrix(lean::Tensor::from(B.center()),
                                                                    lean::Tensor::from(B.rad())));
        }
    });
    return test::finish("lean interval interval_matrix");
}
