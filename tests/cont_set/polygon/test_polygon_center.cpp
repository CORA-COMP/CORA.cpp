// test_polygon_center - Polygon::center: the centroid of the region
//
// CORA: contSet/@polygon/center.m
// PROPERTY: contSet/@polygon/center.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void centroids(const std::string &b) {
    check(close(test::unit_square().center(), {0.5, 0.5}), b + ": the square's middle");
    // the L: (2 * (1, 0.5) + 1 * (0.5, 1.5)) / 3
    check(close(test::l_shape().center(), {2.5 / 3, 2.5 / 3}), b + ": the centroid of the L");
    check(std::isnan(Polygon::empty().center().data()[0]), b + ": NaN for nothing");
}

/// Stated property: a random piece and its copy shifted by v, apart, have the centroid c + v / 2.
void shifted_copy() {
    Rng rng(37);
    for (int trial = 0; trial < 10; ++trial) {
        const Polygon A({test::random_piece(rng, 0, 0)});
        const Tensor c = A.center();
        const Polygon both = A.unite(A.plus(vec({20.0, 0.0})));
        check(close(both.center(), c + vec({10.0, 0.0}), 1e-9), "center: c + v / 2 of two copies");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { centroids(b); });
    shifted_copy();
    {
        ChecksScope off(false);
        check(close(test::unit_square().center(), {0.5, 0.5}),
              "center: the same with the checks off");
    }
    return test::finish("polygon center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
