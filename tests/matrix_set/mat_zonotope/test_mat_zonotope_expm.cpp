// test_mat_zonotope_expm - MatZonotope expm: the matrix exponential, as CORA calls expmInd
//
// CORA: matrixSet/@matZonotope/expm.m
// PROPERTY: matrixSet/@matZonotope/expm.m

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// expm is expm_independent, with the order 10 by default; the exponential of every member is
/// inside.
void expm_is_the_independent_series(const std::string &b) {
    Rng rng(71);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.3);
    check(test::same_set(A.expm(), A.expm_independent(10), 0), b + ": the default order is 10");
    check(test::same_set(A.expm(6), A.expm_independent(6), 0), b + ": the order given");
    const IntervalMatrix box = A.expm().interval_matrix();
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial)
        inside =
            inside && box.contains(test::exponential(test::member(A, test::coefficients(rng, 2))),
                                   "exact", 1e-12);
    check(inside, b + ": exp(M) of every member is inside");
    const MatZonotope point(test::matrix({0, 0, 0, 0}, 2, 2));
    check(close(point.expm().center(), {1, 0, 0, 1}, 1e-12), b + ": exp(0) = I");
}

/// A non-square set and a negative order are refused; with the checks off the same result.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { MatZonotope(test::matrix({1, 2, 3}, 1, 3)).expm(); },
                       Error::dimension_mismatch),
          "expm: a non-square set");
    check(test::raises([&] { A.expm(-1); }, Error::wrong_value), "expm: a negative order");
    const Tensor on = A.expm(3).center();
    ChecksScope off(false);
    check(close(A.expm(3).center(), on, 0), "expm: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(expm_is_the_independent_series);
    checks_and_their_switch();
    return test::finish("mat_zonotope expm");
}
