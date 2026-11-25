// empty - the empty polygon, as CORA's polygon.empty
//
// No piece. CORA's polygon is always two-dimensional, so n must be 2.
//
// Syntax:   Polygon pgon = Polygon::empty();
// Inputs:   n - the dimension, 2 (default)
// Outputs:  pgon - the polygon without a piece
// See also: polygon, representsa
//
// CORA: contSet/@polygon/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polygon Polygon::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n == 2, Error::wrong_value,
               [&] { return "empty: a polygon is two-dimensional, got n = " + std::to_string(n); });
    return Polygon(std::vector<Polytope>{});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
