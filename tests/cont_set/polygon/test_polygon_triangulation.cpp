// test_polygon_triangulation - Polygon::triangulation: triangles without overlap
//
// CORA: contSet/@polygon/triangulation.m
// PROPERTY: contSet/@polygon/triangulation.m
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

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the triangles have three vertices, add up to the area, and do not overlap.
void triangles() {
    Rng rng(29);
    for (int trial = 0; trial < 10; ++trial) {
        const Polygon pgon = test::random_polygon(rng, 3);
        const std::vector<Polytope> T = pgon.triangulation();
        double sum = 0;
        bool three = true;
        for (const Polytope &P : T) sum += P.volume(), three = three && P.V().size(0) == 3;
        check(three && close(sum, pgon.volume(), 1e-9) && close(Polygon(T).volume(), sum, 1e-9),
              "triangulation: triangles of the area, without overlap");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    triangles();
    {
        ChecksScope off(false);
        check(test::l_shape().triangulation().size() >= 3,
              "triangulation: the same with the checks off");
    }
    return test::finish("polygon triangulation");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
