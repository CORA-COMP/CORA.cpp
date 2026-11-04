// test_interval_matrix_exponential_remainder - IntervalMatrix exponential_remainder: the series
// tail
//
// CORA: matrixSet/@intervalMatrix/exponentialRemainder.m
// PROPERTY: matrixSet/@intervalMatrix/exponentialRemainder.m

#include "interval_matrix_testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;

namespace {

/// 1 x 1: A = [-1, 1], |A| = 1, order 2: W = e - (1 + 1 + 1/2); the zero matrix has no remainder.
void remainder_by_hand(const std::string &b) {
    const IntervalMatrix S(test::matrix({0}, 1, 1), test::matrix({1}, 1, 1));
    const double W = std::exp(1.0) - 2.5;
    const Interval E = S.exponential_remainder(2).interval();
    check(close(E.sup().item(), W, 1e-12) && close(E.inf().item(), -W, 1e-12), b + ": [-W, W]");
    const IntervalMatrix zero(test::matrix({0, 0, 0, 0}, 2, 2));
    check(close(zero.exponential_remainder(3).interval().sup(), std::vector<double>{0, 0, 0, 0},
                1e-15),
          b + ": the zero matrix");
}

/// The remainder encloses e^M minus the Taylor sum for every member and shrinks with the order; a
/// matrix with no number as exponential gets Inf; a batch is per member.
void remainder_encloses_the_tail(const std::string &b) {
    Rng rng(16);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.5);
    const IntervalMatrix E = A.exponential_remainder(4);
    bool inside = true;
    for (int trial = 0; trial < 30; ++trial) {
        const Tensor M = test::member(A, rng);
        Tensor power = M.eye_like(3), sum = power;
        double factorial = 1.0;
        for (int i = 1; i <= 4; ++i) {
            power = M.matmul(power);
            factorial *= i;
            sum = sum + power / factorial;
        }
        inside = inside && E.contains(M.expm() - sum, "exact", 1e-12);
    }
    check(inside, b + ": the tail of every member is inside");
    check(A.exponential_remainder(8).interval().sup().max().item() <
              E.interval().sup().max().item(),
          b + ": a higher order has a smaller remainder");
    const IntervalMatrix not_a_number(
        Tensor::from_data({std::numeric_limits<double>::quiet_NaN(), 0, 0, 0}, {2, 2}));
    check(not_a_number.exponential_remainder(3).interval().sup().min().item() ==
              std::numeric_limits<double>::infinity(),
          b + ": Inf where the exponential is not a number");
    const IntervalMatrix T = test::random_matrix(rng, 3, 0.5);
    const IntervalMatrix batch(Interval::stack({A.interval(), T.interval()}));
    check(close(batch.exponential_remainder(4).interval().sup().slice(0, 0, 1).reshape({3, 3}),
                E.interval().sup()),
          b + ": a batch");
}

/// A matrix that is not square and a negative order are refused; the checks off give the same.
void checks_and_their_switch() {
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { wide.exponential_remainder(2); }, Error::dimension_mismatch),
          "exponential_remainder: a matrix that is not square");
    check(test::raises([&] { test::example().exponential_remainder(-1); }, Error::wrong_value),
          "exponential_remainder: a negative order");
    const Tensor on = test::example().exponential_remainder(3).interval().sup();
    ChecksScope off(false);
    check(close(test::example().exponential_remainder(3).interval().sup(), on, 0),
          "exponential_remainder: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        remainder_by_hand(b);
        remainder_encloses_the_tail(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix exponential_remainder");
}
