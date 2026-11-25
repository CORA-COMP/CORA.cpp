// test_polygon_contains - Polygon::contains: points in a vertex triangle of a piece
//
// CORA: contSet/@polygon/contains_.m
// PROPERTY: contSet/@polygon/contains_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: random points of the L are inside iff they lie in [0,2]x[0,1] or [0,1]x[1,2].
void classified() {
    Rng rng(3);
    const Polygon L = test::l_shape();
    const Tensor X = test::random_points(rng, 200, -0.5, 2.5);
    int wrong = 0;
    for (int i = 0; i < 200; ++i) {
        const std::vector<double> x = test::row(X, i).data();
        const bool in =
            (x[0] >= 0 && x[1] >= 0) && ((x[0] <= 2 && x[1] <= 1) || (x[0] <= 1 && x[1] <= 2));
        wrong += L.contains(test::row(X, i)) != in;
    }
    check(wrong == 0, "contains: every random point is classified as the L");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    classified();
    check(test::l_shape().contains(Tensor({{0.5, 1.5}, {1.5, 0.5}})) &&
              !test::l_shape().contains(vec({1.5, 1.5})) &&
              test::unit_square().contains(vec({1.0, 1.0})),
          "contains: inside, outside, a corner");
    check(test::raises([] { test::l_shape().contains(vec({1.0, 0.0, 0.0})); },
                       Error::dimension_mismatch),
          "contains: points of three entries are refused");
    check(
        test::raises([] { test::l_shape().contains(vec({1.0, 0.0}), 0.1); }, Error::not_supported),
        "contains: a tolerance is not supported");
    {
        ChecksScope off(false);
        check(test::l_shape().contains(vec({0.5, 0.5})), "contains: the same with the checks off");
    }
    return test::finish("polygon contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
