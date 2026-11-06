// test_interval_matrix_norm - IntervalMatrix norm: the norm of the largest matrix of the set
//
// CORA: matrixSet/@intervalMatrix/norm.m
// PROPERTY: matrixSet/@intervalMatrix/norm.m
// CORA authors: Matthias Althoff, Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "interval_matrix_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// |center| + rad of the example is [[2, 1], [3, 2]]: column sums 5 and 3, row sums 3 and 5, and
/// the square root of 4 + 1 + 9 + 4.
void norms_of_the_example(const std::string &b) {
    const IntervalMatrix A = test::example();
    check(close(A.norm(Tensor::Norm::one).item(), 5.0), b + ": the 1-norm");
    check(close(A.norm(Tensor::Norm::inf).item(), 5.0), b + ": the Inf-norm");
    check(close(A.norm(Tensor::Norm::fro).item(), std::sqrt(18.0)), b + ": the Frobenius norm");
}

/// No member has a larger norm, and for a set with one sign the bound is reached by a member.
void norm_bounds_the_members(const std::string &b) {
    Rng rng(14);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.8);
    for (const Tensor::Norm type : {Tensor::Norm::one, Tensor::Norm::inf, Tensor::Norm::fro}) {
        const double bound = A.norm(type).item();
        bool below = true;
        for (int trial = 0; trial < 40; ++trial)
            below = below && test::member(A, rng).matrix_norm(type).item() <= bound + 1e-12;
        check(below, b + ": the members' norms are below the norm");
    }
    const IntervalMatrix batch(Interval::stack({A.interval(), A.interval()}));
    check(close(batch.norm(Tensor::Norm::fro).slice(0, 1, 2).reshape({1, 1}),
                A.norm(Tensor::Norm::fro)),
          b + ": a batch");
}

/// The 2-norm is refused, as in CORA; with the checks off a valid norm is the same.
void checks_and_their_switch() {
    const IntervalMatrix A = test::example();
    check(test::raises([&] { A.norm(Tensor::Norm::two); }, Error::wrong_value), "norm: the 2-norm");
    const double on = A.norm(Tensor::Norm::one).item();
    ChecksScope off(false);
    check(close(A.norm(Tensor::Norm::one).item(), on, 0), "norm: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        norms_of_the_example(b);
        norm_bounds_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
