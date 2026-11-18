// test_zonotope_stacked - Zonotope::stacked: the zonotope of the stacked matrices
//
// CORA: matrixSet/@matZonotope/zonotope.m
// PROPERTY: matrixSet/@matZonotope/zonotope.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The example stacks column by column: C = [[1, 2], [3, 4]] gives (1, 3, 2, 4), the generators
/// (1, 0, 0, 1) and (0, 1, 1, 0).
void stacking_of_the_example(const std::string &b) {
    const Zonotope Z = test::example().stacked();
    check(close(Z.center().expand_dims(-1), {1, 3, 2, 4}), b + ": the center, column by column");
    check(Z.generators().matrix_transpose().shape() == std::vector<int64_t>({4, 2}),
          b + ": four dimensions, two generators");
    check(close(Z.generators().matrix_transpose(), {1, 0, 0, 1, 0, 1, 1, 0}),
          b + ": the generators, column by column");
    const Zonotope W = test::point_set(test::matrix({1, 2, 3, 4, 5, 6}, 2, 3)).stacked();
    check(close(W.center().expand_dims(-1), {1, 4, 2, 5, 3, 6}),
          b + ": a non-square matrix, column by column");
    check(W.generators().matrix_transpose().shape() == std::vector<int64_t>({6, 0}),
          b + ": without generators");
}

/// A member's stacking is the zonotope's point for the same coefficients; a batch stacks per set.
void members_map_to_points(const std::string &b) {
    Rng rng(5);
    const Zonotope A = test::random_set(rng, 3, 2, 0.6);
    const Zonotope Z = A.stacked();
    const std::vector<double> coeff = test::coefficients(rng, 2);
    const Tensor point =
        Z.center().expand_dims(-1) + Z.generators().matrix_transpose().matmul(
                                         Tensor::like(Z.center().expand_dims(-1), coeff, {2, 1}));
    const Tensor stacked = test::member(A, coeff).matrix_transpose().reshape({9, 1});
    check(close(point, stacked, 1e-12), b + ": the stacked member is the zonotope's point");
    const Zonotope batch = test::batch_of(A, A.transpose());
    check(batch.stacked().generators().matrix_transpose().shape() ==
              std::vector<int64_t>({2, 9, 2}),
          b + ": a batch");
}

/// zonotope takes no input; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope A = test::example();
    const Tensor on = A.stacked().center();
    ChecksScope off(false);
    check(close(A.stacked().center(), on, 0), "zonotope: the same with the checks off");
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
