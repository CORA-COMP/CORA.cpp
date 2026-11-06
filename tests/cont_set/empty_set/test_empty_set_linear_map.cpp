// test_empty_set_linear_map - EmptySet linear_map: the empty set of the rows of the matrix
//
// CORA: contSet/@emptySet/mtimes.m
// PROPERTY: contSet/@emptySet/mtimes.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_image_has_the_rows_of_the_matrix(const std::string &b) {
    const EmptySet E(3);
    const Tensor M = Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});
    check(E.linear_map(M).dim() == 2 && (M * E).dim() == 2, b + ": a (2, 3) matrix");
    check((2.5 * E).dim() == 3 && E.linear_map(Tensor({{2.0}})).dim() == 3,
          b + ": a number keeps the dimension");
    const Interval IM(M * 0.5, M * 2.0);
    check((IM * E).dim() == 2, b + ": an interval matrix");
    // random matrices of fixed seed: the rows decide
    Rng rng(16);
    for (int rows = 1; rows <= 4; ++rows) {
        std::vector<double> v(static_cast<std::size_t>(rows) * 3);
        rng.normal(v.data(), v.size(), 1.0);
        check(E.linear_map(Tensor::from_data(v, {rows, 3})).dim() == rows,
              b + ": random matrix with " + std::to_string(rows) + " rows");
    }
    check(Set(E.linear_map(M)).is<EmptySet>(), b + ": the result is an EmptySet");
}

/// A matrix with the wrong number of columns is refused; with the checks off the result is the
/// same.
void checks_and_their_switch() {
    const EmptySet E(3);
    check(test::raises([&] { E.linear_map(Tensor({{1.0, 2.0}, {3.0, 4.0}})); },
                       Error::dimension_mismatch),
          "linear_map: a (2, 2) matrix on 3 dimensions");
    check(test::raises([&] { E.linear_map(Tensor::from_data({1.0, 2.0, 3.0}, {3})); },
                       Error::dimension_mismatch),
          "linear_map: a vector, not a matrix");
    ChecksScope off(false);
    check(E.linear_map(Tensor({{1.0, 0.0, 0.0}})).dim() == 1,
          "linear_map: the same with checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_image_has_the_rows_of_the_matrix);
    checks_and_their_switch();
    return test::finish("empty_set linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
