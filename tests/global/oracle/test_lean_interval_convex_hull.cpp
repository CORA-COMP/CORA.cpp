// test_lean_interval_convex_hull - Interval::convex_hull against the CORALean oracle
//
// ORACLE: contSet/@interval/convHull_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.convHull", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            test::oracle::check(I.convex_hull(J), lean::Interval(I).convex_hull(lean::Interval(J)));
        }
    });
    return test::finish("lean interval convex_hull");
}
