// test_lean_interval_matrix_interval_matrix - the constructor IntervalMatrix(C, D) against the
// oracle
//
// ORACLE: matrixSet/@intervalMatrix/intervalMatrix.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.intervalMatrix", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] : {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}}) {
            // the oracle's constructor at a center and a nonnegative radius of exact doubles
            const Interval box = rnd.box_matrix(m, n);
            const IntervalMatrix A(box.center(), box.rad());
            test::oracle::check(A, lean::Interval::interval_matrix(lean::Tensor::from(box.center()),
                                                                   lean::Tensor::from(box.rad())));
        }
    });
    return test::finish("lean interval_matrix constructor");
}
