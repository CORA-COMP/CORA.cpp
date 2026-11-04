// test_interval_matrix_power - interval matrix_power: the power of a square interval matrix
//
// CORA: contSet/@interval/mpower.m
// PROPERTY: contSet/@interval/mpower.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// A = [[[1, 2], [0, 1]], [0, [1, 1]]].
Interval upper_triangular() {
    return Interval(Tensor::from_data({1, 0, 0, 1}, {2, 2}),
                    Tensor::from_data({2, 1, 0, 1}, {2, 2}));
}

void matrix_power_multiplies_the_intervals(const std::string &b) {
    const Interval A = upper_triangular();
    const Interval I0 = A.matrix_power(0);
    check(close(I0.inf(), std::vector<double>{1, 0, 0, 1}) && close(I0.sup(), I0.inf()),
          b + ": the zeroth power is the identity");
    check(close(A.matrix_power(1).sup(), A.sup()) && close(A.matrix_power(1).inf(), A.inf()),
          b + ": the first power is the matrix");
    const Interval A2 = A.matrix_power(2);
    check(close(A2.inf(), std::vector<double>{1, 0, 0, 1}) &&
              close(A2.sup(), std::vector<double>{4, 3, 0, 1}),
          b + ": the square, by the hull of the corner products");
    const Interval A3 = A.matrix_power(3);
    check(close(A3.inf(), std::vector<double>{1, 0, 0, 1}) &&
              close(A3.sup(), std::vector<double>{8, 7, 0, 1}),
          b + ": the cube");
    const Interval B = Interval::stack({A, A}).matrix_power(2);
    check(B.inf().shape() == std::vector<int64_t>({2, 2, 2}) &&
              close(B.sup().slice(0, 1, 2).reshape({2, 2}), A2.sup()),
          b + ": a batch");
}

/// The power of a point matrix is the matrix power, and the result encloses the powers of members.
void matrix_power_of_points_and_samples(const std::string &b) {
    cora::Rng rng(11);
    std::vector<double> m(9);
    rng.normal(m.data(), 9, 1.0);
    const Tensor M = Tensor::from_data(m, {3, 3});
    const Interval P = Interval(M, M).matrix_power(4);
    const Tensor M2 = M.matmul(M);
    check(close(P.inf(), M2.matmul(M2), 1e-9) && close(P.sup(), M2.matmul(M2), 1e-9),
          b + ": a point matrix");
    const Interval A(M - 0.1, M + 0.1);
    const Interval Q = A.matrix_power(3);
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> u(9);
        rng.uniform(u.data(), 9, 0, 1);
        const Tensor X = A.inf() + (A.sup() - A.inf()).multiply(Tensor::from_data(u, {3, 3}));
        const Tensor pow = X.matmul(X).matmul(X);
        inside = inside && (Q.inf().less_equal(pow).min().item() != 0) &&
                 (pow.less_equal(Q.sup()).min().item() != 0);
    }
    check(inside, b + ": the cube of a member is in the result");
}

/// A matrix that is not square, a negative exponent and a scalar base are refused; the checks off
/// give the same.
void checks_and_their_switch() {
    const Interval A = upper_triangular();
    const Interval wide(Tensor::from_data({1, 2, 3}, {1, 3}), Tensor::from_data({1, 2, 3}, {1, 3}));
    check(test::raises([&] { wide.matrix_power(2); }, Error::wrong_value),
          "matrix_power: a matrix that is not square");
    check(test::raises([&] { A.matrix_power(-1); }, Error::wrong_value),
          "matrix_power: a negative exponent");
    const Interval scalar(Tensor::from_data({1}, {1, 1}), Tensor::from_data({2}, {1, 1}));
    check(test::raises([&] { scalar.matrix_power(2); }, Error::not_supported),
          "matrix_power: a 1 x 1 base");
    const Tensor on = A.matrix_power(3).sup();
    ChecksScope off(false);
    check(close(A.matrix_power(3).sup(), on, 0), "matrix_power: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        matrix_power_multiplies_the_intervals(b);
        matrix_power_of_points_and_samples(b);
    });
    checks_and_their_switch();
    return test::finish("interval matrix_power");
}
