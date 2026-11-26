// test_polygon_is_full_dim - Polygon::is_full_dim: some piece has an area
//
// CORA: contSet/@polygon/isFullDim.m
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

void interior() {
    check(test::l_shape().is_full_dim(), "is_full_dim: the L");
    check(!Polygon({Polytope(Tensor({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}}))}).is_full_dim(),
          "is_full_dim: a segment has none");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    interior();
    {
        ChecksScope off(false);
        check(test::l_shape().is_full_dim(), "is_full_dim: the same with the checks off");
    }
    return test::finish("polygon is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
