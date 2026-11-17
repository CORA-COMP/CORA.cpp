// test_mat_exp_expm_norm_infinity - Interval expm_norm_infinity: CORA's closed form
//
// CORA: matrixSet/@intervalMatrix/expmNormInf.m
// PROPERTY: matrixSet/@intervalMatrix/expmNormInf.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// By hand for C = 0.2, D = 0.1, t = 1: nA = 0.2, nS = 0.1, nAS = 0.3, so
/// (0.01 / 0.765) + (0.1 / 0.85) = 0.130718954...; a point matrix and t = 0 give 0.
void value_of_the_formula(const std::string &b) {
    const Interval A = test::interval_matrix(test::matrix({0.2}, 1, 1), test::matrix({0.1}, 1, 1));
    check(close(mat_exp::expm_norm_infinity(A, 1.0), std::vector<double>{0.01 / 0.765 + 0.1 / 0.85},
                1e-12),
          b + ": 1 x 1 by hand");
    check(close(mat_exp::expm_norm_infinity(A, 0.0), std::vector<double>{0.0}), b + ": t = 0");
    check(close(mat_exp::expm_norm_infinity(
                    test::interval_matrix(test::matrix({0.3, -0.1, 0, 0.2}, 2, 2)), 1.0),
                std::vector<double>{0.0}),
          b + ": a point matrix has no radius");
}

/// Wider sets give larger bounds (below the pole of the formula); a batch is per member.
void bound_grows_with_the_radius(const std::string &b) {
    Rng rng(32);
    const Tensor C = test::matrix({0.1, -0.05, 0.02, 0.08}, 2, 2);
    const Interval thin = test::interval_matrix(C, test::matrix({0.01, 0.01, 0.01, 0.01}, 2, 2));
    const Interval wide = test::interval_matrix(C, test::matrix({0.05, 0.05, 0.05, 0.05}, 2, 2));
    check(mat_exp::expm_norm_infinity(thin, 1.0).item() <
              mat_exp::expm_norm_infinity(wide, 1.0).item(),
          b + ": monotone");
    const Interval T = test::random_matrix(rng, 2, 0.05);
    const Interval batch = Interval::stack({wide, T});
    const Tensor v = mat_exp::expm_norm_infinity(batch, 1.0);
    check(v.shape() == std::vector<int64_t>{2, 1, 1} &&
              close(v.slice(0, 0, 1).reshape({1}),
                    mat_exp::expm_norm_infinity(wide, 1.0).reshape({1}), 0),
          b + ": a batch");
}

/// A negative time is refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    const Interval A = test::interval_matrix(test::matrix({0.2}, 1, 1), test::matrix({0.1}, 1, 1));
    check(test::raises([&] { mat_exp::expm_norm_infinity(A, -1.0); }, Error::wrong_value),
          "expm_norm_infinity: a negative time");
    const Tensor on = mat_exp::expm_norm_infinity(A, 1.0);
    ChecksScope off(false);
    check(close(mat_exp::expm_norm_infinity(A, 1.0), on, 0),
          "expm_norm_infinity: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        value_of_the_formula(b);
        bound_grows_with_the_radius(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_norm_infinity");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
