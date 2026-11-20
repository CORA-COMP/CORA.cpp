// test_mat_exp_zonotope_expm_one_param - the Taylor series of a matrix zonotope of one generator
//
// CORA: matrixSet/@matZonotope/expmOneParam.m
// PROPERTY: matrixSet/@matZonotope/expmOneParam.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The coefficients of the generators for the parameter b: b^l for odd l and 2 b^l - 1 for even l,
/// whose half is moved to the center.
std::vector<double> generator_coefficients(double b, int64_t order) {
    std::vector<double> c;
    for (int64_t l = 1; l <= order; ++l)
        c.push_back(l % 2 ? std::pow(b, l) : 2 * std::pow(b, l) - 1);
    return c;
}

/// sum_{k=0}^N r^k M^k / k! of the matrix M.
Tensor taylor(const Tensor &M, double r, int64_t N) {
    Tensor sum = M.eye_like(M.size(-1)) + M.zeros_like(), power = sum;
    double factorial = 1.0;
    for (int64_t k = 1; k <= N; ++k) {
        power = power.matmul(M);
        factorial *= double(k);
        sum = sum + power * (std::pow(r, double(k)) / factorial);
    }
    return sum;
}

/// A 1 x 1 set M = 1 + b at r = 1 and N = 2: K = 2.75 + 2 b + 0.25 (2 b^2 - 1); CORA starts the
/// sum of b^2 at the coefficient of b in M^2 and gets 3 + 2 b + 0.5 (2 b^2 - 1).
void scalar_example(const std::string &b) {
    const Zonotope A = test::set_of({1}, {1}, 1, 1);
    const auto R = mat_exp::expm_one_param(A, 1.0, 2);
    check(close(R.enclosure.center(), {2.75}) && close(R.enclosure.generators(), {2, 0.25}),
          b + ": K = 2.75 + 2 b + 0.25 (2 b^2 - 1)");
    check(R.powers.size() == 2 && close(R.powers[0].center(), {1}) &&
              close(R.powers[0].generators(), {1}),
          b + ": the first power is A r");
    check(close(R.powers[1].center(), {1}) && close(R.powers[1].generators(), {2, 1}),
          b + ": the second power, 1 + 2 b + b^2");
    check(!R.constant_input.has_value(), b + ": no input without one");
    // M = b at r = 1, N = 2: 1 + b + b^2 / 2 reaches 2.5; exp(1) is in K + remainder, CORA misses
    const Zonotope M = Zonotope::matrix(test::matrix({0}, 1, 1), Tensor::from_data({1}, {1, 1, 1}));
    const auto S = mat_exp::expm_one_param(M, 1.0, 2);
    check(close(S.enclosure.center(), {1.25}) && close(S.enclosure.generators(), {1, 0.25}),
          b + ": K = 1.25 + b + 0.25 (2 b^2 - 1)");
    const Interval total = S.enclosure.interval() + S.remainder;
    check(total.contains(test::matrix({std::exp(1.0)}, 1, 1), 1e-12), b + ": exp(1) is enclosed");
}

/// For every parameter b in [-1, 1] the Taylor polynomial of r M(b) is the matrix of K for the
/// coefficients b^l and 2 b^l - 1; the powers are the powers of r M(b) in b; the remainder box
/// contains what the series leaves out, so exp(r M(b)) is in K + remainder.
void series_for_one_parameter(const std::string &bname) {
    Rng rng(91);
    for (const int64_t N : {1, 2, 5, 8}) {
        const Zonotope A = test::random_set(rng, 3, 1, 0.5);
        const double r = 0.6;
        const auto R = mat_exp::expm_one_param(A, r, N);
        check(R.enclosure.nr_generators() == N && int64_t(R.powers.size()) == N,
              bname + ": N generators and N powers, N = " + std::to_string(N));
        bool exact = true, powers_exact = true, enclosed = true;
        for (const double b : {-1.0, -0.4, 0.0, 0.7, 1.0}) {
            const Tensor M = test::member(A, {b});
            exact = exact && close(test::member(R.enclosure, generator_coefficients(b, N)),
                                   taylor(M, r, N), 1e-12);
            for (int64_t k = 1; k <= N; ++k) {
                std::vector<double> powers_of_b;
                for (int64_t l = 1; l <= k; ++l) powers_of_b.push_back(std::pow(b, l));
                Tensor power = M;
                for (int64_t j = 1; j < k; ++j) power = power.matmul(M);
                // the exact powers
                powers_exact = powers_exact && close(test::member(R.powers[k - 1], powers_of_b),
                                                     power * std::pow(r, double(k)), 1e-12);
            }
            const Tensor rest = test::exponential(M * r) - taylor(M, r, N);
            enclosed = enclosed && R.remainder.contains(rest, 1e-12);
        }
        check(exact, bname + ": the Taylor polynomial is K for the coefficients, N = " +
                         std::to_string(N));
        check(powers_exact, bname + ": the powers, N = " + std::to_string(N));
        check(enclosed, bname + ": the remainder bounds the rest, N = " + std::to_string(N));
    }
}

/// The input terms sum_{k=1}^N r^k M^(k-1) u / k! for u(b) = c + b g are the point of the zonotope
/// for the same coefficients; a column is the zonotope without extent.
void series_with_an_input(const std::string &bname) {
    Rng rng(92);
    const Zonotope A = test::random_set(rng, 3, 1, 0.5);
    const double r = 0.5;
    const int64_t N = 6;
    const Tensor c = test::matrix({1, -2, 0.5}, 3, 1), g = test::matrix({0.3, 0.1, -0.4}, 3, 1);
    const auto R = mat_exp::expm_one_param(A, r, N, Zonotope(c.squeeze(-1), g.matrix_transpose()));
    check(R.constant_input.has_value() &&
              R.constant_input->generators().matrix_transpose().shape() ==
                  std::vector<int64_t>({3, N}),
          bname + ": the input's zonotope has N generators");
    // for every parameter value: the series of the member, term by term
    bool exact = true;
    for (const double b : {-1.0, 0.3, 1.0}) {
        const Tensor M = test::member(A, {b}), u = c + g * b;
        Tensor expected = u * r, power = u;
        double factorial = 1.0;
        for (int64_t k = 2; k <= N; ++k) {
            power = M.matmul(power);
            factorial *= double(k);
            expected = expected + power * (std::pow(r, double(k)) / factorial);
        }
        // a point of the constant input
        const Zonotope &Z = *R.constant_input;
        std::vector<double> coeff = generator_coefficients(b, N);
        const Tensor point = Z.center().expand_dims(-1) +
                             Z.generators().matrix_transpose().matmul(
                                 Tensor::like(Z.center().expand_dims(-1), coeff, {N, 1}));
        exact = exact && close(point, expected, 1e-12);
    }
    check(exact, bname + ": the input terms for every parameter");
    // an input without generators gives the same sets as one with a zero generator
    const auto fixed =
        mat_exp::expm_one_param(A, r, N, Zonotope(c.squeeze(-1), c.zeros_like({1, 3})));
    const auto zero_generators =
        mat_exp::expm_one_param(A, r, N, Zonotope(c.squeeze(-1), Tensor::zeros({0, 3})));
    check(close(zero_generators.constant_input->center().expand_dims(-1),
                fixed.constant_input->center().expand_dims(-1), 0) &&
              close(zero_generators.constant_input->generators().matrix_transpose(),
                    fixed.constant_input->generators().matrix_transpose(), 0) &&
              test::same_set(fixed.enclosure, R.enclosure, 0),
          bname + ": a zonotope without generators is a column");
    // a batch gives each member's series
    const Zonotope batch = test::batch_of(A, A);
    check(test::same_set(test::batch_member(mat_exp::expm_one_param(batch, r, 4).enclosure, 1),
                         mat_exp::expm_one_param(A, r, 4).enclosure, 1e-12),
          bname + ": a batch");
}

/// More or fewer than one generator, a non-square set, an order below 1, an input of another
/// dimension or with several generators are refused; with the checks off the same result.
void checks_and_their_switch() {
    const Zonotope A = test::set_of({1, 0, 0, 1}, {0, 1, 0, 0}, 2, 2);
    check(
        test::raises([&] { mat_exp::expm_one_param(test::example(), 1.0, 3); }, Error::wrong_value),
        "expm_one_param: two generators");
    check(test::raises([&] { mat_exp::expm_one_param(test::point_set(A.center()), 1.0, 3); },
                       Error::wrong_value),
          "expm_one_param: no generator");
    // a set of non-square matrices and an order below 1
    const Zonotope wide = Zonotope::matrix(test::matrix({1, 2, 3}, 1, 3), Tensor::zeros({1, 1, 3}));
    check(test::raises([&] { mat_exp::expm_one_param(wide, 1.0, 3); }, Error::dimension_mismatch),
          "expm_one_param: a non-square set");
    check(test::raises([&] { mat_exp::expm_one_param(A, 1.0, 0); }, Error::wrong_value),
          "expm_one_param: the order 0");
    const Zonotope too_long(test::vec({1, 2, 3}), test::matrix({0, 0, 0}, 1, 3));
    const Zonotope two(test::vec({1, 2}), test::matrix({1, 0, 0, 1}, 2, 2));
    check(test::raises([&] { mat_exp::expm_one_param(A, 1.0, 2, too_long); },
                       Error::dimension_mismatch),
          "expm_one_param: an input of another dimension");
    check(test::raises([&] { mat_exp::expm_one_param(A, 1.0, 2, two); }, Error::not_supported),
          "expm_one_param: an input with two generators");
    const Tensor on = mat_exp::expm_one_param(A, 1.0, 3).enclosure.center();
    ChecksScope off(false);
    // the same with the checks off
    check(close(mat_exp::expm_one_param(A, 1.0, 3).enclosure.center(), on, 0),
          "expm_one_param: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        scalar_example(b);
        series_for_one_parameter(b);
        series_with_an_input(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope expm_one_param");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
