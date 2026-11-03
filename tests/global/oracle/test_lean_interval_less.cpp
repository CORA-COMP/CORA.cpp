// test_lean_interval_less - Interval::less against the CORALean oracle (its `lt`)
//
// The oracle's strict comparison adds the margin 2^-64 rounding up, which at the dtype nearest is
// absorbed: it reads equal bounds as strictly inside. So the exact mode leaves out the box against
// itself, which the sound mode checks.
//
// ORACLE: contSet/@interval/lt.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.lt", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n), big = I.enlarge(2.0);
            test::oracle::check(I.less(big), lean::Interval(I).lt(lean::Interval(big)));
            test::oracle::check(big.less(I), lean::Interval(big).lt(lean::Interval(I)));
            test::oracle::check(I.less(J), lean::Interval(I).lt(lean::Interval(J)));
            if (test::oracle::mode() == test::oracle::Mode::sound)
                test::oracle::check(I.less(I), lean::Interval(I).lt(lean::Interval(I)));
        }
    });
    return test::finish("lean interval less");
}
