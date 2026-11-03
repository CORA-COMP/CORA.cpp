// test_lean_interval_enclose_points - Interval::enclose_points against the CORALean oracle
//
// ORACLE: contSet/@interval/enclosePoints.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.enclosePoints", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5}) {
            const Tensor points = rnd.matrix(n, 7);
            test::oracle::check(Interval::enclose_points(points),
                                lean::Interval::enclose_points(lean::Tensor::from(points)));
        }
    });
    return test::finish("lean interval enclose_points");
}
