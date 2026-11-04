// test_lean_interval_matrix_matmul - IntervalMatrix::matmul against the CORALean oracle
//
// The product with an interval matrix is the oracle's mtimes; the product with a matrix of numbers
// is the same with a point matrix.
//
// ORACLE: matrixSet/@intervalMatrix/mtimes.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.mtimes", [](test::oracle::Random &rnd) {
        for (const auto &[m, n, p] :
             {std::tuple{2, 2, 2}, std::tuple{3, 4, 2}, std::tuple{1, 5, 3}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n)), B(rnd.box_matrix(n, p));
            const lean::IntervalMatrix LA{lean::Interval(A.interval())},
                LB{lean::Interval(B.interval())};
            test::oracle::check(A.matmul(B), LA.matmul(LB.interval()));
            const Tensor M = rnd.matrix(n, p), N = rnd.matrix(p, m);
            const lean::Interval point_M = lean::Interval::from(M, M);
            test::oracle::check(A.matmul(M), LA.matmul(point_M));
            const lean::IntervalMatrix point_N{lean::Interval::from(N, N)};
            test::oracle::check(IntervalMatrix::matmul(N, A), point_N.matmul(LA.interval()));
        }
    });
    return test::finish("lean interval_matrix matmul");
}
