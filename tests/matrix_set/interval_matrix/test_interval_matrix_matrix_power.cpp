// test_interval_matrix_matrix_power - IntervalMatrix matrix_power: the power of a square set
//
// CORA: matrixSet/@intervalMatrix/mpower.m
// PROPERTY: matrixSet/@intervalMatrix/mpower.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// A = [[[1, 2], [0, 1]], [0, 1]]: the powers by the hull of the corner products.
void powers_of_a_triangular_matrix(const std::string &b) {
    const IntervalMatrix A(Interval(test::matrix({1, 0, 0, 1}, 2, 2), test::matrix({2, 1, 0, 1}, 2, 2)));
    const Interval P0 = A.matrix_power(0).interval();
    check(close(P0.inf(), std::vector<double>{1, 0, 0, 1}) && close(P0.sup(), P0.inf()), b + ": A^0 = I");
    check(close(A.matrix_power(1).interval().sup(), A.interval().sup()), b + ": A^1 = A");
    check(close(A.matrix_power(2).interval().sup(), std::vector<double>{4, 3, 0, 1}), b + ": A^2");
    check(close(A.matrix_power(3).interval().sup(), std::vector<double>{8, 7, 0, 1}), b + ": A^3");
    // the product of intervals, not the power of an interval: [-1, 1]^2 is [-1, 1] here
    const IntervalMatrix S(Interval(test::matrix({-1}, 1, 1), test::matrix({1}, 1, 1)));
    check(close(S.matrix_power(2).interval().inf(), std::vector<double>{-1}), b + ": a 1 x 1 matrix");
}

/// The powers of members are in the powers of the set, and a point matrix is its power.
void powers_enclose_the_members(const std::string &b) {
    Rng rng(11);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.2);
    for (const int k : {2, 3, 4}) {
        const IntervalMatrix P = A.matrix_power(k);
        bool inside = true;
        for (int trial = 0; trial < 30; ++trial) {
            Tensor M = test::member(A, rng), power = M;
            for (int i = 1; i < k; ++i) power = power.matmul(M);
            inside = inside && P.contains(power, "exact", 1e-12);
        }
        check(inside, b + ": members, k = " + std::to_string(k));
    }
    const Tensor M = test::member(A, rng);
    check(close(IntervalMatrix(M).matrix_power(3).center(), M.matmul(M).matmul(M), 1e-12),
          b + ": a point matrix");
}

/// A matrix that is not square and a negative exponent are refused; the checks off give the same.
void checks_and_their_switch() {
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { wide.matrix_power(2); }, Error::dimension_mismatch),
          "matrix_power: a matrix that is not square");
    check(test::raises([&] { test::example().matrix_power(-1); }, Error::wrong_value),
          "matrix_power: a negative exponent");
    const Tensor on = test::example().matrix_power(3).interval().sup();
    ChecksScope off(false);
    check(close(test::example().matrix_power(3).interval().sup(), on, 0),
          "matrix_power: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        powers_of_a_triangular_matrix(b);
        powers_enclose_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix matrix_power");
}
