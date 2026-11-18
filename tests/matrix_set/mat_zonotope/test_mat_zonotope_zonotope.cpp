// test_mat_zonotope_zonotope - MatZonotope zonotope: the zonotope of the stacked matrices
//
// CORA: matrixSet/@matZonotope/zonotope.m
// PROPERTY: matrixSet/@matZonotope/zonotope.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The example stacks column by column: C = [[1, 2], [3, 4]] gives (1, 3, 2, 4), the generators
/// (1, 0, 0, 1) and (0, 1, 1, 0).
void stacking_of_the_example(const std::string &b) {
    const Zonotope Z = test::example().zonotope();
    // layout: #163 step 2
    check(close(detail::cora_center(Z), {1, 3, 2, 4}), b + ": the center, column by column");
    // layout: #163 step 2
    check(detail::cora_generators(Z).shape() == std::vector<int64_t>({4, 2}),
          b + ": four dimensions, two generators");
    // layout: #163 step 2
    check(close(detail::cora_generators(Z), {1, 0, 0, 1, 0, 1, 1, 0}),
          b + ": the generators, column by column");
    const Zonotope W = MatZonotope(test::matrix({1, 2, 3, 4, 5, 6}, 2, 3)).zonotope();
    // layout: #163 step 2
    check(close(detail::cora_center(W), {1, 4, 2, 5, 3, 6}),
          b + ": a non-square matrix, column by column");
    // layout: #163 step 2
    check(detail::cora_generators(W).shape() == std::vector<int64_t>({6, 0}),
          b + ": without generators");
}

/// A member's stacking is the zonotope's point for the same coefficients; a batch stacks per set.
void members_map_to_points(const std::string &b) {
    Rng rng(5);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.6);
    const Zonotope Z = A.zonotope();
    const std::vector<double> coeff = test::coefficients(rng, 2);
    const Tensor point =
        // layout: #163 step 2
        detail::cora_center(Z) +
        detail::cora_generators(Z).matmul(Tensor::like(detail::cora_center(Z), coeff, {2, 1}));
    const Tensor stacked = test::member(A, coeff).matrix_transpose().reshape({9, 1});
    check(close(point, stacked, 1e-12), b + ": the stacked member is the zonotope's point");
    const MatZonotope batch = test::batch_of(A, A.transpose());
    // layout: #163 step 2
    check(detail::cora_generators(batch.zonotope()).shape() == std::vector<int64_t>({2, 9, 2}),
          b + ": a batch");
}

/// zonotope takes no input; with the checks off the result is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    // layout: #163 step 2
    const Tensor on = detail::cora_center(A.zonotope());
    ChecksScope off(false);
    // layout: #163 step 2
    check(close(detail::cora_center(A.zonotope()), on, 0),
          "zonotope: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        stacking_of_the_example(b);
        members_map_to_points(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
