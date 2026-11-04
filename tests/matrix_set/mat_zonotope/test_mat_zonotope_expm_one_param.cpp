// test_mat_zonotope_expm_one_param - MatZonotope expm_one_param: the Taylor series for one
// parameter
//
// CORA: matrixSet/@matZonotope/expmOneParam.m
// PROPERTY: matrixSet/@matZonotope/expmOneParam.m

#include "mat_zonotope_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

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

/// A 1 x 1 set M = 1 + b at r = 1 and N = 2: 1 + M + M^2 / 2 = 2.5 + 2 b + b^2 / 2, so the
/// generator of b^2 is 0.5 (halved: 0.25, and 0.25 more in the center): K = 2.75 + 2 b + 0.25 (2
/// b^2 - 1). CORA starts the sum of b^2 at the coefficient of b in M^2 and gets 3 + 2 b + 0.5 (2
/// b^2 - 1).
void scalar_example(const std::string &b) {
    const MatZonotope A = test::set_of({1}, {1}, 1, 1);
    const auto R = A.expm_one_param(1.0, 2);
    check(close(R.enclosure.center(), {2.75}) && close(R.enclosure.generators(), {2, 0.25}),
          b + ": K = 2.75 + 2 b + 0.25 (2 b^2 - 1)");
    check(R.powers.size() == 2 && close(R.powers[0].center(), {1}) &&
              close(R.powers[0].generators(), {1}),
          b + ": the first power is A r");
    check(close(R.powers[1].center(), {1}) && close(R.powers[1].generators(), {2, 1}),
          b + ": the second power, 1 + 2 b + b^2");
    check(!R.constant_input.has_value(), b + ": no input without one");
    // M = b at r = 1, N = 2: 1 + b + b^2 / 2 reaches 2.5 and exp(1) is in K + remainder; CORA's sum
    // gives 1 + b, so its K + remainder ends at 2.218 < exp(1)
    const MatZonotope M(test::matrix({0}, 1, 1), Tensor::from_data({1}, {1, 1, 1}));
    const auto S = M.expm_one_param(1.0, 2);
    check(close(S.enclosure.center(), {1.25}) && close(S.enclosure.generators(), {1, 0.25}),
          b + ": K = 1.25 + b + 0.25 (2 b^2 - 1)");
    const IntervalMatrix total(S.enclosure.interval_matrix().interval() + S.remainder.interval());
    check(total.contains(test::matrix({std::exp(1.0)}, 1, 1), "exact", 1e-12),
          b + ": exp(1) is enclosed");
}

/// For every parameter b in [-1, 1] the Taylor polynomial of r M(b) is the matrix of K for the
/// coefficients b^l and 2 b^l - 1; the powers are the powers of r M(b) in b; the remainder box
/// contains what the series leaves out, so exp(r M(b)) is in K + remainder.
void series_for_one_parameter(const std::string &bname) {
    Rng rng(91);
    for (const int64_t N : {1, 2, 5, 8}) {
        const MatZonotope A = test::random_set(rng, 3, 1, 0.5);
        const double r = 0.6;
        const auto R = A.expm_one_param(r, N);
        check(R.enclosure.num_generators() == N && int64_t(R.powers.size()) == N,
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
                powers_exact = powers_exact && close(test::member(R.powers[k - 1], powers_of_b),
                                                     power * std::pow(r, double(k)), 1e-12);
            }
            const Tensor rest = test::exponential(M * r) - taylor(M, r, N);
            enclosed = enclosed && R.remainder.contains(rest, "exact", 1e-12);
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
    const MatZonotope A = test::random_set(rng, 3, 1, 0.5);
    const double r = 0.5;
    const int64_t N = 6;
    const Tensor c = test::matrix({1, -2, 0.5}, 3, 1), g = test::matrix({0.3, 0.1, -0.4}, 3, 1);
    const auto R = A.expm_one_param(r, N, Zonotope(c, g));
    check(R.constant_input.has_value() &&
              R.constant_input->generators().shape() == std::vector<int64_t>({3, N}),
          bname + ": the input's zonotope has N generators");
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
        const Zonotope &Z = *R.constant_input;
        std::vector<double> coeff = generator_coefficients(b, N);
        const Tensor point =
            Z.center() + Z.generators().matmul(Tensor::like(Z.center(), coeff, {N, 1}));
        exact = exact && close(point, expected, 1e-12);
    }
    check(exact, bname + ": the input terms for every parameter");
    const auto fixed = A.expm_one_param(r, N, Zonotope(c, c.zeros_like()));
    const auto zero_generators = A.expm_one_param(r, N, Zonotope(c, Tensor::zeros({3, 0})));
    check(close(zero_generators.constant_input->center(), fixed.constant_input->center(), 0) &&
              close(zero_generators.constant_input->generators(),
                    fixed.constant_input->generators(), 0) &&
              test::same_set(fixed.enclosure, R.enclosure, 0),
          bname + ": a zonotope without generators is a column");
    const MatZonotope batch = test::batch_of(A, A);
    check(test::same_set(test::batch_member(batch.expm_one_param(r, 4).enclosure, 1),
                         A.expm_one_param(r, 4).enclosure, 1e-12),
          bname + ": a batch");
}

/// More or fewer than one generator, a non-square set, an order below 1, an input of another
/// dimension or with several generators are refused; with the checks off the same result.
void checks_and_their_switch() {
    const MatZonotope A = test::set_of({1, 0, 0, 1}, {0, 1, 0, 0}, 2, 2);
    check(test::raises([&] { test::example().expm_one_param(1.0, 3); }, Error::wrong_value),
          "expm_one_param: two generators");
    check(test::raises([&] { MatZonotope(A.center()).expm_one_param(1.0, 3); }, Error::wrong_value),
          "expm_one_param: no generator");
    const MatZonotope wide(test::matrix({1, 2, 3}, 1, 3), Tensor::zeros({1, 1, 3}));
    check(test::raises([&] { wide.expm_one_param(1.0, 3); }, Error::dimension_mismatch),
          "expm_one_param: a non-square set");
    check(test::raises([&] { A.expm_one_param(1.0, 0); }, Error::wrong_value),
          "expm_one_param: the order 0");
    const Zonotope too_long(test::matrix({1, 2, 3}, 3, 1), test::matrix({0, 0, 0}, 3, 1));
    const Zonotope two(test::matrix({1, 2}, 2, 1), test::matrix({1, 0, 0, 1}, 2, 2));
    check(test::raises([&] { A.expm_one_param(1.0, 2, too_long); }, Error::dimension_mismatch),
          "expm_one_param: an input of another dimension");
    check(test::raises([&] { A.expm_one_param(1.0, 2, two); }, Error::not_supported),
          "expm_one_param: an input with two generators");
    const Tensor on = A.expm_one_param(1.0, 3).enclosure.center();
    ChecksScope off(false);
    check(close(A.expm_one_param(1.0, 3).enclosure.center(), on, 0),
          "expm_one_param: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        scalar_example(b);
        series_for_one_parameter(b);
        series_with_an_input(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope expm_one_param");
}
