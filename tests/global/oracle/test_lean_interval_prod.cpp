// test_lean_interval_prod - Interval::prod against the CORALean oracle (its `prod`)
//
// The product of the rows of an interval matrix is a row, in the oracle (from batch 5) and in
// CORA.cpp. The oracles before gave a column; with them the matrix is not compared.
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
        const lean::Interval theirs = lean::Interval(M).prod();
        if (theirs.inf().rows() == 1)
            test::oracle::check(M.prod(1), theirs);
        else
            test::oracle::skipped(
                "the oracle's interval.prod of a matrix is a column (older oracle)");
    });
    return test::finish("lean interval prod");
}
