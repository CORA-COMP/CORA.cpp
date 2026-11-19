// test_mat_exp_dependent_terms - Interval dependent_terms: r^2 M^2 and I + M r + (M
//
// r)^2 / 2
//
// CORA: matrixSet/@intervalMatrix/dependentTerms.m
// PROPERTY: matrixSet/@intervalMatrix/dependentTerms.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A point matrix M gives r^2 M^2 and I + r M + r^2 M^2 / 2 exactly: M = [[1, 2], [3, 4]].
void terms_of_a_point_matrix(const std::string &b) {
    const Interval M = test::interval_matrix(test::matrix({1, 2, 3, 4}, 2, 2));
    const auto [sq1, H1] = mat_exp::dependent_terms(M, 1.0);
    check(close(sq1.center(), std::vector<double>{7, 10, 15, 22}), b + ": r = 1, the square");
    check(close(H1.center(), std::vector<double>{5.5, 7, 10.5, 16}), b + ": r = 1, H");
    const auto [sq2, H2] = mat_exp::dependent_terms(M, 0.5);
    check(close(sq2.center(), std::vector<double>{1.75, 2.5, 3.75, 5.5}),
          b + ": r = 0.5, the square");
    check(close(H2.center(), std::vector<double>{2.375, 2.25, 3.375, 5.75}), b + ": r = 0.5, H");
    check(close(H2.half_width(), std::vector<double>{0, 0, 0, 0}), b + ": a point stays a point");
}

/// The terms of every member M are in the results; the diagonal range includes the vertex -1/r.
void terms_enclose_the_members(const std::string &b) {
    Rng rng(10);
    for (const double r : {1.0, 0.5, 0.125}) {
        const Interval A = test::random_matrix(rng, 3, 0.7);
        const auto [sq, H] = mat_exp::dependent_terms(A, r);
        bool inside = true;
        for (int trial = 0; trial < 40; ++trial) {
            const Tensor M = test::member(A, rng), I = M.eye_like(3);
            const Tensor M2 = M.matmul(M) * (r * r);
            const Tensor T = I + M * r + M.matmul(M) * (0.5 * r * r);
            inside = inside && sq.contains(M2, 1e-12) && H.contains(T, 1e-12);
        }
        check(inside, b + ": members inside, r = " + std::to_string(r));
    }
    // -1/r inside the diagonal entry: the lower end of H is I - 1/2 there
    const Interval V = Interval::matrix(test::matrix({-3}, 1, 1), test::matrix({-1}, 1, 1));
    check(close(mat_exp::dependent_terms(V, 1.0).second.infimum(), std::vector<double>{0.5}, 1e-9),
          b + ": the vertex of the parabola");
}

/// A matrix that is not square is refused; with the checks off a valid one is the same.
void checks_and_their_switch() {
    const Interval wide = test::interval_matrix(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { mat_exp::dependent_terms(wide, 1.0); }, Error::dimension_mismatch),
          "dependent_terms: a matrix that is not square");
    const Tensor on = mat_exp::dependent_terms(test::example(), 0.5).second.center();
    ChecksScope off(false);
    check(close(mat_exp::dependent_terms(test::example(), 0.5).second.center(), on, 0),
          "dependent_terms: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        terms_of_a_point_matrix(b);
        terms_enclose_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix dependent_terms");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
