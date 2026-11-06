// test_interval_matrix_infinity_norm - IntervalMatrix infinity_norm: the largest row sum of abs
//
// CORA: matrixSet/@intervalMatrix/infNorm.m
// PROPERTY: matrixSet/@intervalMatrix/infNorm.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// abs of the example is [[2, 1], [3, 2]]: the row sums are 3 and 5.
void infinity_norm_of_the_example(const std::string &b) {
    check(close(test::example().infinity_norm().item(), 5.0), b + ": the largest row sum");
    check(close(IntervalMatrix(test::matrix({-1, -2, 3, 0}, 2, 2)).infinity_norm().item(), 3.0),
          b + ": a point matrix");
}

/// No member has a larger Inf-norm, and the bound is the Inf-norm of abs; a batch is per member.
void norm_bounds_the_members(const std::string &b) {
    Rng rng(15);
    const IntervalMatrix A = test::random_matrix(rng, 4, 1.0);
    const double bound = A.infinity_norm().item();
    bool below = true;
    for (int trial = 0; trial < 40; ++trial)
        below =
            below && test::member(A, rng).matrix_norm(Tensor::Norm::inf).item() <= bound + 1e-12;
    check(below, b + ": the members' Inf-norms are below it");
    check(close(bound, A.abs().matrix_norm(Tensor::Norm::inf).item()), b + ": the Inf-norm of abs");
    const IntervalMatrix T = test::random_matrix(rng, 2, 1.0);
    const IntervalMatrix batch(Interval::stack({T.interval(), test::example().interval()}));
    check(close(batch.infinity_norm().slice(0, 1, 2).reshape({1, 1}),
                test::example().infinity_norm()),
          b + ": a batch");
}

/// infinity_norm has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const double on = test::example().infinity_norm().item();
    ChecksScope off(false);
    check(close(test::example().infinity_norm().item(), on, 0),
          "infinity_norm: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        infinity_norm_of_the_example(b);
        norm_bounds_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix infinity_norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
