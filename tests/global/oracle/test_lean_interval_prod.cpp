// test_lean_interval_prod - Interval::prod against the CORALean oracle (its `prod`)
//
// The oracle gives the product of the rows of an interval matrix as a column, CORA.cpp as a row.
//
// ORACLE: contSet/@interval/prod.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.prod", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 12}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.prod(), lean::Interval(I).prod());
        }
        const Tensor c = rnd.matrix(6, 3), r = Tensor::from_data({0.5, 0.25, 1.0}, {1, 3});
        const Interval M(c - r, c + r);
        const Interval row = M.prod(1);
        test::oracle::check(Interval(row.inf().matrix_transpose(), row.sup().matrix_transpose()),
                            lean::Interval(M).prod());
    });
    return test::finish("lean interval prod");
}
