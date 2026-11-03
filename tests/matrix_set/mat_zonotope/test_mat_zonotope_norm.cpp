// test_mat_zonotope_norm - MatZonotope norm: an upper bound of the norm of the set
//
// CORA: matrixSet/@matZonotope/norm.m
// PROPERTY: matrixSet/@matZonotope/norm.m

#include "mat_zonotope_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

/// |center| + rad of the example is [[2, 3], [4, 5]]: column sums 6 and 8, row sums 5 and 9, and
/// the square root of 4 + 9 + 16 + 25.
void norms_of_the_example(const std::string &b) {
    const MatZonotope A = test::example();
    check(close(A.norm(Tensor::Norm::one).item(), 8.0), b + ": the 1-norm");
    check(close(A.norm(Tensor::Norm::inf).item(), 9.0), b + ": the Inf-norm");
    check(close(A.norm(Tensor::Norm::fro).item(), std::sqrt(54.0)), b + ": the Frobenius norm");
    check(close(A.norm(Tensor::Norm::fro).item(),
                A.interval_matrix().norm(Tensor::Norm::fro).item(), 0),
          b + ": the norm of the box around the set");
}

/// No member has a larger norm; a batch gives one bound per set.
void norm_bounds_the_members(const std::string &b) {
    Rng rng(15);
    const MatZonotope A = test::random_set(rng, 3, 3, 0.8);
    for (const Tensor::Norm type : {Tensor::Norm::one, Tensor::Norm::inf, Tensor::Norm::fro}) {
        const double bound = A.norm(type).item();
        bool below = true;
        for (int trial = 0; trial < 40; ++trial)
            below = below && test::member(A, test::coefficients(rng, 3)).matrix_norm(type).item() <=
                                 bound + 1e-12;
        check(below, b + ": the members' norms are below the norm");
    }
    const MatZonotope batch = test::batch_of(A, A);
    check(close(batch.norm(Tensor::Norm::fro).slice(0, 1, 2).reshape({1, 1}),
                A.norm(Tensor::Norm::fro)),
          b + ": a batch");
}

/// The 2-norm is refused, as in CORA; with the checks off a valid norm is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { A.norm(Tensor::Norm::two); }, Error::wrong_value), "norm: the 2-norm");
    const double on = A.norm(Tensor::Norm::one).item();
    ChecksScope off(false);
    check(close(A.norm(Tensor::Norm::one).item(), on, 0), "norm: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        norms_of_the_example(b);
        norm_bounds_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope norm");
}
