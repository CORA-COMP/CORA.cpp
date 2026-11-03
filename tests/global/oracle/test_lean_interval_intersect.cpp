// test_lean_interval_intersect - Interval::intersect against the CORALean oracle (its `and`)
//
// ORACLE: contSet/@interval/and_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.and", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // J is I shifted by a third of its width, so the boxes overlap
            const Interval I = rnd.box(n);
            const Tensor shift = (I.sup() - I.inf()) * 0.3;
            const Interval J(I.inf() + shift, I.sup() + shift);
            test::oracle::check(I.intersect(J), lean::Interval(I).intersect(lean::Interval(J)));
        }
    });
    return test::finish("lean interval intersect");
}
