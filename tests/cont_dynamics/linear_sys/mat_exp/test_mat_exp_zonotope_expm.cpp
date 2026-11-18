// test_mat_exp_zonotope_expm - the exponential of a matrix zonotope, as CORA calls expmInd
//
// CORA: matrixSet/@matZonotope/expm.m
// PROPERTY: matrixSet/@matZonotope/expm.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// expm is expm_independent, with the order 10 by default; the exponential of every member is
/// inside.
void expm_is_the_independent_series(const std::string &b) {
    Rng rng(71);
    const Zonotope A = test::random_set(rng, 3, 2, 0.3);
    check(test::same_set(mat_exp::expm(A), mat_exp::expm_independent(A, 10), 0),
          b + ": the default order is 10");
    check(test::same_set(mat_exp::expm(A, 6), mat_exp::expm_independent(A, 6), 0),
          b + ": the order given");
    const Interval box = mat_exp::expm(A).interval();
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial)
        inside =
            inside &&
            box.contains(test::exponential(test::member(A, test::coefficients(rng, 2))), 1e-12);
    check(inside, b + ": exp(M) of every member is inside");
    const Zonotope point = test::point_set(test::matrix({0, 0, 0, 0}, 2, 2));
    check(close(mat_exp::expm(point).center(), {1, 0, 0, 1}, 1e-12), b + ": exp(0) = I");
}

/// A non-square set and a negative order are refused; with the checks off the same result.
void checks_and_their_switch() {
    const Zonotope A = test::example();
    check(test::raises([&] { mat_exp::expm(test::point_set(test::matrix({1, 2, 3}, 1, 3))); },
                       Error::dimension_mismatch),
          "expm: a non-square set");
    check(test::raises([&] { mat_exp::expm(A, -1); }, Error::wrong_value),
          "expm: a negative order");
    const Tensor on = mat_exp::expm(A, 3).center();
    ChecksScope off(false);
    check(close(mat_exp::expm(A, 3).center(), on, 0), "expm: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(expm_is_the_independent_series);
    checks_and_their_switch();
    return test::finish("mat_zonotope expm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
