// test_polygon_rand_point - Polygon::rand_point: uniform points of the region
//
// CORA: contSet/@polygon/randPoint_.m
// PROPERTY: contSet/@polygon/randPoint_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the points lie in the polygon, and the share in a part follows its area.
void uniform() {
    Rng rng(31);
    const Polygon L = test::l_shape();
    const Tensor X = L.rand_point(3000, rng);
    int upper = 0;
    for (int i = 0; i < 3000; ++i) upper += test::row(X, i).data()[1] > 1;
    check(X.shape() == Shape{3000, 2} && L.contains(X), "rand_point: (N, 2) points of the L");
    check(std::abs(upper / 3000.0 - 1.0 / 3) < 0.05, "rand_point: a third above y = 1");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    uniform();
    Rng rng(1);
    check(test::raises([&] { Polygon::empty().rand_point(3, rng); }, Error::empty_set),
          "rand_point: the empty polygon has no points");
    {
        ChecksScope off(false);
        check(test::unit_square().rand_point(5, rng).shape() == Shape{5, 2},
              "rand_point: the same with the checks off");
    }
    return test::finish("polygon rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
