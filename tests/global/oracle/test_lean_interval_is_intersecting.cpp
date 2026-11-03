// test_lean_interval_is_intersecting - Interval::is_intersecting against the CORALean oracle
//
// ORACLE: contSet/@interval/isIntersecting_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.isIntersecting", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n), K = I.enlarge(0.5);
            test::oracle::check(I.is_intersecting(J, 0),
                                lean::Interval(I).is_intersecting(lean::Interval(J), 0));
            test::oracle::check(I.is_intersecting(K, 0),
                                lean::Interval(I).is_intersecting(lean::Interval(K), 0));
            test::oracle::check(I.is_intersecting(I, 0),
                                lean::Interval(I).is_intersecting(lean::Interval(I), 0));
        }
    });
    return test::finish("lean interval is_intersecting");
}
