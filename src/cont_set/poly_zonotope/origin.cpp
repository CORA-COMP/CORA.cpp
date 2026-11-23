// origin - the polynomial zonotope {0}, as CORA's polyZonotope.origin
//
// Syntax:   PolyZonotope pZ = PolyZonotope::origin(n);
// Inputs:   n - dimension, n >= 1
// Outputs:  pZ - the set with center 0 (n) and no generators
// See also: Zonotope::origin
//
// CORA: contSet/@polyZonotope/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS PolyZonotope PolyZonotope::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value,
               [&] { return "origin: the dimension must be positive, got " + std::to_string(n); });
    return PolyZonotope(Zonotope::origin(n));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
