// test_mat_exp_exponential_remainder - Interval exponential_remainder: the series
//
// tail
//
// CORA: matrixSet/@intervalMatrix/exponentialRemainder.m
// PROPERTY: matrixSet/@intervalMatrix/exponentialRemainder.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// 1 x 1: A = [-1, 1], |A| = 1, order 2: W = e - (1 + 1 + 1/2); the zero matrix has no remainder.
void remainder_by_hand(const std::string &b) {
    const Interval S = test::interval_matrix(test::matrix({0}, 1, 1), test::matrix({1}, 1, 1));
    const double W = std::exp(1.0) - 2.5;
    const Interval E = mat_exp::exponential_remainder(S, 2);
    check(close(E.sup().item(), W, 1e-12) && close(E.inf().item(), -W, 1e-12), b + ": [-W, W]");
    const Interval zero = test::interval_matrix(test::matrix({0, 0, 0, 0}, 2, 2));
    check(close(mat_exp::exponential_remainder(zero, 3).sup(), std::vector<double>{0, 0, 0, 0},
                1e-15),
          b + ": the zero matrix");
}

/// The remainder encloses e^M minus the Taylor sum for every member and shrinks with the order; a
/// matrix with no number as exponential gets Inf; a batch is per member.
void remainder_encloses_the_tail(const std::string &b) {
    Rng rng(16);
    const Interval A = test::random_matrix(rng, 3, 0.5);
    const Interval E = mat_exp::exponential_remainder(A, 4);
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
        inside = inside && E.contains(M.expm() - sum, 1e-12);
    }
    check(inside, b + ": the tail of every member is inside");
    check(mat_exp::exponential_remainder(A, 8).sup().max().item() < E.sup().max().item(),
          b + ": a higher order has a smaller remainder");
    // a NaN matrix has an infinite remainder
    const Interval not_a_number = test::interval_matrix(
        Tensor::from_data({std::numeric_limits<double>::quiet_NaN(), 0, 0, 0}, {2, 2}));
    check(mat_exp::exponential_remainder(not_a_number, 3).sup().min().item() ==
              std::numeric_limits<double>::infinity(),
          b + ": Inf where the exponential is not a number");
    const Interval T = test::random_matrix(rng, 3, 0.5);
    const Interval batch = Interval::stack({A, T});
    check(close(mat_exp::exponential_remainder(batch, 4).sup().slice(0, 0, 1).reshape({3, 3}),
                E.sup()),
          b + ": a batch");
}

/// A matrix that is not square and a negative order are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval wide = test::interval_matrix(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { mat_exp::exponential_remainder(wide, 2); }, Error::dimension_mismatch),
          "exponential_remainder: a matrix that is not square");
    check(test::raises([&] { mat_exp::exponential_remainder(test::example(), -1); },
                       Error::wrong_value),
          "exponential_remainder: a negative order");
    const Tensor on = mat_exp::exponential_remainder(test::example(), 3).sup();
    ChecksScope off(false);
    check(close(mat_exp::exponential_remainder(test::example(), 3).sup(), on, 0),
          "exponential_remainder: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        remainder_by_hand(b);
        remainder_encloses_the_tail(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix exponential_remainder");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
