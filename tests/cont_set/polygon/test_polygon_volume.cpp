// test_polygon_volume - Polygon::volume: the area of the region
//
// CORA: contSet/@polygon/volume_.m
// PROPERTY: contSet/@polygon/volume_.m
// CORA authors: Tobias Ladner
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

/// Stated property: the area of a union is the sum of the pieces' areas less their overlap.
void inclusion_exclusion() {
    Rng rng(17);
    for (int trial = 0; trial < 20; ++trial) {
        const Polygon A({test::random_piece(rng, 0, 0)}), B({test::random_piece(rng, 0.5, 0)});
        const double overlap = A.intersect(B).volume();
        check(close(A.unite(B).volume(), A.volume() + B.volume() - overlap, 1e-9),
              "volume: |A u B| = |A| + |B| - |A n B|");
        check(close(A.volume(), A.pieces()[0].volume(), 1e-9), "volume: one piece is its area");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    inclusion_exclusion();
    check(close(test::l_shape().volume(), 3.0) && Polygon::empty().volume() == 0,
          "volume: the L, nothing");
    {
        ChecksScope off(false);
        check(close(test::l_shape().volume(), 3.0), "volume: the same with the checks off");
    }
    return test::finish("polygon volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
