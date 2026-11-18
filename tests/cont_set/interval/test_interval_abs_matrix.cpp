// test_interval_abs_matrix - Interval abs: the entrywise largest absolute value
//
// CORA: matrixSet/@intervalMatrix/abs.m
// PROPERTY: matrixSet/@intervalMatrix/abs.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void abs_of_the_example(const std::string &b) {
    check(close(test::example().abs().sup(), std::vector<double>{2, 1, 3, 2}),
          b + ": max(|inf|, |sup|)");
    const Interval S =
        test::interval_matrix(test::matrix({-3, 1}, 1, 2), test::matrix({0, 0}, 1, 2));
    check(close(S.abs().sup(), std::vector<double>{3, 1}), b + ": a point matrix");
}

/// Every member is below abs entrywise, and the bound is reached at an end; a batch is per member.
void abs_bounds_the_members(const std::string &b) {
    Rng rng(3);
    const Interval A = test::random_matrix(rng, 3, 1.0);
    bool below = true;
    for (int trial = 0; trial < 30; ++trial)
        below =
            below && test::member(A, rng).abs().less_equal(A.abs().sup() + 1e-15).min().item() != 0;
    check(below, b + ": |M| <= abs(A) for members");
    check(close(A.abs().sup(), A.infimum().abs().maximum(A.sup().abs())),
          b + ": reached at an end");
    const Interval T = test::random_matrix(rng, 2, 1.0);
    const Interval batch = Interval::stack({T, test::example()});
    check(close(batch.abs().sup().slice(0, 1, 2).reshape({2, 2}), test::example().abs().sup()),
          b + ": a batch");
}

/// abs has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor on = test::example().abs().sup();
    ChecksScope off(false);
    check(close(test::example().abs().sup(), on, 0), "abs: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        abs_of_the_example(b);
        abs_bounds_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix abs");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
