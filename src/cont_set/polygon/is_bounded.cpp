// is_bounded - whether a polygon is bounded, as CORA's isBounded
//
// Always: a finite union of bounded pieces.
//
// Syntax:   bool b = pgon.is_bounded();
// Inputs:   -
// Outputs:  b - true
// See also: is_full_dim
//
// CORA: contSet/@polygon/isBounded.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Polygon::is_bounded_impl() const { return true; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
