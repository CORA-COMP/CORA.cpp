// test_polygon_vertices - Polygon::vertices: one counter-clockwise outline per piece
//
// CORA: contSet/@polygon/vertices_.m
// PROPERTY: contSet/@polygon/vertices_.m
// CORA authors: Tobias Ladner
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

/// Stated property: every outline turns left only, and its points are stored vertices.
void outlines() {
    Rng rng(13);
    const Polygon pgon = test::random_polygon(rng, 3);
    const std::vector<Outline> loops = pgon.vertices();
    bool left = loops.size() == 3;
    for (const Outline &o : loops)
        for (std::size_t i = 0; i < o.size(); ++i) {
            const Point a = o[i], b = o[(i + 1) % o.size()], c = o[(i + 2) % o.size()];
            left = left && (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0;
        }
    check(left, "vertices: counter-clockwise convex outlines, one per piece");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    outlines();
    check(test::raises([] { test::l_shape().vertices_3d(); }, Error::not_supported),
          "vertices: a polygon has no three-dimensional vertices");
    {
        ChecksScope off(false);
        check(test::unit_square().vertices().size() == 1, "vertices: the same with the checks off");
    }
    return test::finish("polygon vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
