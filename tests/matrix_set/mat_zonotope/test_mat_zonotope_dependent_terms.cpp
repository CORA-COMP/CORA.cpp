// test_mat_zonotope_dependent_terms - MatZonotope dependent_terms: the square and the second-order
//
// terms
//
// CORA: matrixSet/@matZonotope/dependentTerms.m
// PROPERTY: matrixSet/@matZonotope/dependentTerms.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The coefficients of the generators of the square: b_i, 2 b_i^2 - 1 and b_i b_j for i < j.
std::vector<double> square_coefficients(const std::vector<double> &b) {
    std::vector<double> c(b);
    for (const double bi : b) c.push_back(2 * bi * bi - 1);
    for (std::size_t i = 0; i < b.size(); ++i)
        for (std::size_t j = i + 1; j < b.size(); ++j) c.push_back(b[i] * b[j]);
    return c;
}

/// The matrix of the set for generator coefficients c, whatever their number.
Tensor combine(const MatZonotope &A, const std::vector<double> &c) { return test::member(A, c); }

/// A 1 x 1 set M = 2 + b at r = 1: M^2 = 4 + 4b + b^2 = 4.5 + 4b + (2b^2 - 1) / 2 and
/// I + M + M^2 / 2 = 5.25 + 3b + 0.25 (2b^2 - 1).
void scalar_example(const std::string &b) {
    const MatZonotope A = test::set_of({2}, {1}, 1, 1);
    const auto [sq, H] = A.dependent_terms(1.0);
    check(close(sq.center(), {4.5}) && close(sq.generators(), {4, 0.5}), b + ": the square");
    check(close(H.center(), {5.25}) && close(H.generators(), {3, 0.25}),
          b + ": I + M r + M^2 r^2 / 2");
    const auto [sq2, H2] = A.dependent_terms(0.5);
    check(close(sq2.center(), {1.125}) && close(H2.center(), {1 + 1.0 + 0.5625}),
          b + ": the step 0.5");
}

/// For every member M the square r^2 M^2 and the series I + M r + M^2 r^2 / 2 are the sets'
/// matrices for the coefficients b_i, 2 b_i^2 - 1 and b_i b_j; the numbers of generators are 2h +
/// h(h - 1) / 2.
void members_are_enclosed(const std::string &b) {
    Rng rng(51);
    for (const int64_t h : {1, 2, 3}) {
        const MatZonotope A = test::random_set(rng, 3, h, 0.5);
        const double r = 0.7;
        const auto [sq, H] = A.dependent_terms(r);
        check(sq.num_generators() == 2 * h + h * (h - 1) / 2 &&
                  H.num_generators() == sq.num_generators(),
              b + ": the number of generators, h = " + std::to_string(h));
        bool exact = true;
        for (int trial = 0; trial < 10; ++trial) {
            const std::vector<double> coeff = test::coefficients(rng, h),
                                      c = square_coefficients(coeff);
            const Tensor M = test::member(A, coeff);
            exact =
                exact && close(combine(sq, c), M.matmul(M) * (r * r), 1e-12) &&
                close(combine(H, c), M.eye_like(3) + M * r + M.matmul(M) * (0.5 * r * r), 1e-12);
        }
        check(exact, b + ": the matrices of the members, h = " + std::to_string(h));
    }
    // a batch gives the same terms
    Rng again(52);
    const MatZonotope A = test::random_set(again, 2, 2, 0.5);
    const auto [sq, H] = test::batch_of(A, A).dependent_terms(0.5);
    check(test::same_set(test::batch_member(H, 1), A.dependent_terms(0.5).second, 1e-12),
          b + ": a batch");
    const auto [point_sq, point_H] = MatZonotope(A.center()).dependent_terms(0.5);
    const Tensor square = A.center().matmul(A.center()) * 0.25;
    check(point_sq.num_generators() == 0 && close(point_sq.center(), square, 1e-12),
          b + ": a point matrix");
}

/// A non-square set is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { MatZonotope(test::matrix({1, 2, 3}, 1, 3)).dependent_terms(1.0); },
                       Error::dimension_mismatch),
          "dependent_terms: a non-square set");
    const Tensor on = A.dependent_terms(0.5).second.center();
    ChecksScope off(false);
    check(close(A.dependent_terms(0.5).second.center(), on, 0),
          "dependent_terms: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        scalar_example(b);
        members_are_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope dependent_terms");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
